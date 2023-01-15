# %%
import os
import random
import numpy as np
import pandas as pd
import pyabc
from pygam import LinearGAM  # For graphing posteriors
from pyabc.transition.multivariatenormal import MultivariateNormalTransition  # For drawing from the posterior
import matplotlib.pyplot as plt
import plotly.express as px  # Mostly use plotly express where possible
import plotly.graph_objects as go  # Sometimes need fine-grained control
from plotly.subplots import make_subplots
# For plotting plotly in spyder. Opens in a web browser
from plotly.offline import plot

from btc_model import BTCModel
from btc_model import Action

pyabc.settings.set_figure_params('pyabc')  # for beautified plots
pd.options.display.max_columns = 160
pd.set_option('max_colwidth', 1000)

DO_ALL_PLOTS = True
ABC_RESUME = False  # Whether to load the most recent run from the dabase, or start a new one

if __name__ == "__main__":
    # %% read data

    print("Preparing data ... ", end="", flush=True)
    # data from: https://www.kaggle.com/mczielinski/bitcoin-historical-data
    df_raw = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv.zip")
    df_raw = df_raw.dropna()
    df_raw['datetime'] = pd.to_datetime(df_raw.Timestamp, unit='s')
    df_raw = df_raw.set_index('datetime', drop=True)  # DateTime index allows for easy resampling


    # Start from 2018, after that first big jump in BTC price (otherwise the model can
    # just buy BTC early on and then hold it
    # df = df.loc['2018-01-01':]

    # Temporarily just look at small time window
    # df = df.loc['2018-01-01':'2018-03-01']

    # %% Resample and filter time series

    def get_first_last(series, first_or_last):
        """Return first or last item in a series, checking for an empty series"""
        if len(series) == 0:
            return np.NaN
        if first_or_last == "first":
            return series.head(1).values[0]
        elif first_or_last == "last":
            return series.tail(1).values[0]
        raise Exception(f"Should not have got here. Series: {series}")


    # Resample, but no filtering yet
    df_resample = df_raw.resample('10min').agg({
        'Timestamp': np.mean, 'Open': lambda x: get_first_last(x, "first"), 'High': np.max,
        'Low': np.min, 'Close': lambda x: get_first_last(x, "last"),
        'Volume_(BTC)': np.sum, 'Volume_(Currency)': np.sum,
        'Weighted_Price': np.mean
    }).dropna()

    # Calculate diff column (different between openning and closing)
    df_resample['Diff'] = df_resample.Close - df_resample.Open

    # Filter dates. Choose a time where there wasn't a big increase to encourage the algorithm to buy and sell a lot.
    # Optimise the models for this time period and then run on a longer time period later
    df = df_resample.loc['2018-01-01':'2019-12-31']

    # Plot BTC price change
    if DO_ALL_PLOTS:
        fig = px.line(df_resample, x=df_resample.index,
                      y=['Open', 'High', 'Low', 'Close', 'Diff'],
                      title='BTC (all variables)')
        plot(fig)

    print("... data prepared")

    # %% Run two models
    print("Running two example models ... ", end="", flush=True)
    btc_historic_price = list(df.Close)  # Pass closing price as the price history
    m1 = BTCModel(btc_historic_price=btc_historic_price, do_history=True)  # Use defaults
    m2 = BTCModel(btc_historic_price=btc_historic_price, do_history=True,
                  buy_action_pct=0.10)  # Increase the action percentage

    portfolio1 = m1.run()  # (this is the same as history.Portfolio_Balance)
    history1 = m1.history.to_df()
    portfolio2 = m2.run()  # (this is the same as history.Portfolio_Balance)
    history2 = m2.history.to_df()

    assert len(history1) == len(df)

    # Sanity check for the two different ways to run models (can use 'run()' or call the
    # model directory (via the BTCModel.__call__() function)
    history3 = m2({"buy_action_pct": 0.05})  # Second model, but now has same parameters as m1
    assert np.equal(m1.history.portfolio_balance, history3.portfolio_balance).all()

    # %% Plot model performance

    # Create figures with secondary y-axis (for the portfolio)
    if DO_ALL_PLOTS:
        fig = make_subplots(specs=[[{"secondary_y": True}, {"secondary_y": True}]], rows=1, cols=2,
                            subplot_titles=(f"Model 1. Performance: {BTCModel.performance(portfolio1)}",
                                            f"Model 2. Performance: {BTCModel.performance(portfolio2)}"))
        for i, hist in enumerate([history1, history2]):
            fig.add_trace(
                go.Scatter(x=df.index, y=hist.Portfolio_Balance, mode='lines', name='portfolio',
                           line={'color': 'black'}),
                secondary_y=False, row=1, col=i + 1)
            fig.add_trace(
                go.Scatter(x=df.index, y=hist.Cash_Balance, mode='lines', name='cash', line={'color': 'lightblue'}),
                secondary_y=False, row=1, col=i + 1)
            fig.add_trace(
                go.Scatter(x=df.index, y=hist.BTC_Balance * 1000, mode='lines', name='btc',
                           line={'color': 'darkblue'}),
                secondary_y=False, row=1, col=i + 1)
            fig.add_trace(
                go.Scatter(x=df.index, y=df.Close, name='BTC price', line={'color': 'orange'}),
                secondary_y=True, row=1, col=i + 1)
        fig.update_layout(title_text="Portfolio performance")
        fig.update_xaxes(title_text="xaxis title")
        fig.update_yaxes(title_text="<b>portfolio</b> performance", secondary_y=False)
        fig.update_yaxes(title_text="<b>BTC</b> price", secondary_y=True)
        plot(fig)

    print("\t ... finished running preliminary models", flush=True)

    # %% Simulation Based Inference

    # Define proiors
    #buy_action_pct_rv = pyabc.RV("norm", 0.05, 0.1)
    #sell_action_pct_rv = pyabc.RV("norm", 0.05, 0.1)
    buy_action_pct_rv = pyabc.RV("norm", 0.0, 0.1)
    sell_action_pct_rv = pyabc.RV("norm", 0.0, 0.1)
    buy_time_window_rv = pyabc.RV("norm", 5, 20)
    sell_time_window_rv = pyabc.RV("norm", 5, 20)

    # Plot them to check they look OK
    if DO_ALL_PLOTS:
        fig = make_subplots(rows=1, cols=2)
        x = np.linspace(-0, 0.5, 100)
        fig.add_trace(
            go.Scatter(x=x, y=pyabc.Distribution(param=buy_action_pct_rv).pdf({"param": x}),
                       mode='lines', name='buy action perecentage', opacity=0.5), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=x, y=pyabc.Distribution(param=sell_action_pct_rv).pdf({"param": x}),
                       mode='lines', name='sell action perecentage', opacity=0.5), row=1, col=1)
        x = np.linspace(-0, 100, 100)
        fig.add_trace(
            go.Scatter(x=x, y=pyabc.Distribution(param=buy_time_window_rv).pdf({"param": x}),
                       mode='lines', name='time window (buying)', opacity=0.5), row=1, col=2)
        fig.add_trace(
            go.Scatter(x=x, y=pyabc.Distribution(param=sell_time_window_rv).pdf({"param": x}),
                       mode='lines', name='time window (selling)', opacity=0.5), row=1, col=2)
        plot(fig)

    # Decorate the RVs so that they wont go below 0 and create the prior distribution
    # The names of the variables in the distribution must match those that the
    # BTCModel constructor is expecting
    # Note, could have created the distribution first and then plotted, e.g.:
    #   y= priors['action_pct'].pdf(x)
    # but decorating them with the LowerBoundDecorator breaks the call to pdf() (?)
    priors = pyabc.Distribution(
        #buy_action_pct=pyabc.LowerBoundDecorator(buy_action_pct_rv, 0.0),
        #sell_action_pct=pyabc.LowerBoundDecorator(sell_action_pct_rv, 0.0),
        buy_action_pct=buy_action_pct_rv,
        sell_action_pct=sell_action_pct_rv,
        buy_time_window=pyabc.LowerBoundDecorator(buy_time_window_rv, 0.0),
        sell_time_window=pyabc.LowerBoundDecorator(sell_time_window_rv, 0.0),
    )

    # %% Run ABC
    print("Preparing ABC ... ", flush=True)


    def btc_model_distance(sim, obs, prevent_negatives=True):
        """Quantify the performance of a model ('sim') by comparing it to some benchmark ('obs').
        The difference is obs-sim, so positive numbers indicate the sim is not as good as obs (expected)
        :param obs: A benchmark ideal portfolio performance (a value for the portfolio at every iteration) as a dictionary
            with a 'data' key (this is how the results are returned by BTCModel.run_abc().
        :param sim: same as obs, but data generated by a BTCModel
        :param prevent_negatives: Prevent negative numbers (i.e. the sim doing better than
            the hypothetical observations) by returning 0 rather than a negative difference. Default: true."""
        # Use BTC.performance
        # obs_performance = BTCModel.performance(obs["data"])  # Convert the array into an overal 'performance'
        # sim_performance = BTCModel.performance(sim["data"])
        # diff2 = obs_performance - sim_performance  # (called diff2 because there is 'diff' in outer scope
        # Take mean of the difference in every iteration
        diff2 = np.mean(obs["data"] - sim["data"])
        if diff2 < 0 and prevent_negatives:
            return 0
        else:
            return diff2


    # Need to generate some observations that represent excellent performance.
    # Make a perfect dataset that goes up with the BTC price, but doesn't come down again. Index on 100
    # (similar to starting with a £100 investment)
    start = df.at[df.index[0], 'Close']
    y_observed = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            y_observed[0] = 100  # Start with £100
        else:
            new_scaled_price = (df.at[df.index[i], 'Close'] / start) * 100  # (to index on 100)
            old_scale_price = (df.at[df.index[i - 1], 'Close'] / start) * 100  # (to index on 100)
            diff = new_scaled_price - old_scale_price
            if diff <= 0:  # If price drops then just keep the old price
                y_observed[i] = y_observed[i - 1]
            else:  # Otherwise increase the value of the portfolio by the difference
                y_observed[i] = y_observed[i - 1] + diff

    # £x at every iteration.
    # y_observed = [100000 for _ in range(len(df_resample))]

    if DO_ALL_PLOTS:
        fig = px.line(x=df.index, y=y_observed, title='Observation data (perfect portfolio performance)')
        plot(fig)

    # 'Template' model to be called in ABC. Useful for setting constant parameters
    abc_model_template = BTCModel(btc_historic_price=btc_historic_price, do_history=False)
    abc_model_template.__name__ = 'template_model'  # Hack to stop pyABC breaking.
    abc = pyabc.ABCSMC(
        models=abc_model_template,  # Model (could be a list). Note BTCModel.__call__ function is called
        parameter_priors=priors,  # Priors (could be a list)
        distance_function=btc_model_distance,  # Distance function defined earlier
        sampler=pyabc.sampler.MulticoreParticleParallelSampler(n_procs=int(os.cpu_count() / 2))
        # sampler=pyabc.sampler.MulticoreEvalParallelSampler()  # The default sampler
        # sampler=pyabc.sampler.SingleCoreSampler()  # Single core for testing
    )

    db_path = ("sqlite:///" + os.path.join(".", "btc_abc.db"))
    # See if there are already some runs in that database. If so, then all_history.all_runs()
    # will return a list with > 0 elements in
    old_history = pyabc.History(db_path)
    run_id = -1

    if ABC_RESUME and len(old_history.all_runs()) > 0:  # Try to load the most recent run
        run_id = old_history.all_runs()[-1].id
        print(f"Loading ABC run {run_id} from the database... ", flush=True)
        abc_history = abc.load(db_path, run_id)
    else:
        run_id = abc.new(db_path, {"data": y_observed})  # (ID only matters if multiple runs stored is same DB)
        print(f"Running new ABC with id {run_id}.... ", flush=True)
        abc_history = abc.run(max_nr_populations=20, minimum_epsilon=1.0)
        print("\t ... ABC run completed.")
    assert run_id > 0

    # %% Algorithm diagnostics

    _, arr_ax = plt.subplots(2, 2)
    pyabc.visualization.plot_sample_numbers(abc_history, ax=arr_ax[0][0])
    pyabc.visualization.plot_epsilons(abc_history, ax=arr_ax[0][1])
    # pyabc.visualization.plot_credible_intervals(
    #    history, levels=[0.95, 0.9, 0.5], ts=[0, 1, 2, 3, 4],
    #    show_mean=True, show_kde_max_1d=True,
    #    refval={'mean': 2.5},
    #    arr_ax=arr_ax[1][0])
    pyabc.visualization.plot_effective_sample_sizes(abc_history, ax=arr_ax[1][1])

    plt.gcf().set_size_inches((12, 8))
    plt.gcf().tight_layout()
    plt.show()

    # %% Marginal posteriors
    fig, axes = plt.subplots(2, int(len(priors) / 2) + 1, figsize=(12, 8))

    for i, param in enumerate(priors.keys()):
        ax = axes.flat[i]
        for t in range(abc_history.max_t + 1):
            df_raw_, w = abc_history.get_distribution(m=0, t=t)
            pyabc.visualization.plot_kde_1d(df_raw_, w, x=param, ax=ax,
                                            label=f"{param} PDF t={t}",
                                            alpha=1.0 if t == 0 else float(t) / abc_history.max_t,
                                            # Make earlier populations transparent
                                            color="black" if t == abc_history.max_t else None  # Make the last one black
                                            )
            # ax.legend()
            ax.set_title(f"{param}")
    fig.tight_layout()
    fig.show()

    # %% 2D correlations
    pyabc.visualization.plot_histogram_matrix(abc_history, size=(12, 10))
    plt.show()

    # %% Analyse the posterior
    # Look at the particles in the last iteration, ordered by weight
    _df, _w = abc_history.get_distribution(m=0, t=abc_history.max_t)
    # Merge dataframe and weights and sort by weight (highest weight at the top)
    _df['weight'] = _w
    posterior_df = _df.sort_values('weight', ascending=False).reset_index()
    print(posterior_df.to_markdown())

    # %% Sample from the posterior
    # Sample from the distribution of parameter posteriors to generate a distribution over the
    # most likely model results. Use kernel density approximation to randomly draw some equally
    # weighted samples.
    print("Sampling from the posterior...", end="", flush=True)
    N_samples = 100
    dist_df, dist_w = abc_history.get_distribution(m=0, t=abc_history.max_t)

    # Sample from the dataframe of posteriors using KDE
    kde = MultivariateNormalTransition(scaling=1)
    kde.fit(dist_df, dist_w)
    samples = kde.rvs(N_samples)

    # Now run N models and store the results of each one
    sample_histories = []

    for i, sample in samples.iterrows():

        # Check for negatives. May need to resample
        if True in [param < 0 for param in sample]:
            print(f"WARNING Found negatives in sample {i}: \n{sample}")
            continue
            # sample = kde.rvs()

        # Create a dictionary with the parameters and their values for this sample
        param_values = {param: sample[str(param)] for param in priors}

        # Run the model. Create a template again (same as before but this time with history)
        template_model = BTCModel(btc_historic_price=btc_historic_price, do_history=True)
        model_hist = template_model(param_values)
        # print(f"Performance: {BTCModel.performance(model_hist.portfolio_balance)}.")
        sample_histories.append(model_hist)
    print("... finished sampling.")

    # %% Plot the sampling results
    print("Plotting samples ... ", end="", flush=True)
    model_histories_df = pd.DataFrame(
        {f"Model {i}": hist.portfolio_balance for (i, hist) in enumerate(sample_histories)}
    )
    assert len(model_histories_df) == len(df)
    # model_histories_df['timestamp'] = df_resample.index
    model_histories_df.index = df.index
    # temporarily filter, too big to display!
    # model_histories_df = model_histories_df.loc['2018-01-01':]
    # temporarily aggregate to daily count
    model_histories_plottable_df = model_histories_df.resample('1D').mean()

    fig = go.Figure()
    for i, col in enumerate(model_histories_plottable_df.columns):
        fig.add_trace(go.Scatter(
            x=model_histories_plottable_df.index,
            y=model_histories_plottable_df[col].values,
            showlegend=False if i > 0 else True,  # hides trace name from legend
            name=None if i > 0 else "Individual models",
            hoverinfo='skip',  # turns off hoverinfo
            mode='lines', line_color='black',
            opacity=2 / len(model_histories_plottable_df.columns)
        ))
    # Add mean
    fig.add_trace(go.Scatter(
        x=model_histories_plottable_df.index,
        y=model_histories_plottable_df.mean(axis=1).to_frame("mean").iloc[:, 0],
        name='Mean', line_color='red', line=dict(width=1),
        mode='lines'))
    fig.update_layout(title_text="Model predictions posterior")

    # Uses plotly express, simpler but no opacity :-(
    # fig = px.line(model_histories_plottable_df, x=model_histories_plottable_df.index,
    #              y=[col for col in model_histories_plottable_df.columns],
    #              title='Posterior samples')
    plot(fig)
    print("..finished plotting")

    # %% Run four optimal models over a longer time period not just the time period it was optimised for
    df_test = df_resample.loc['2015-01-01':]
    btc_historic_price_test = df_test.Close

    print("Running optimal models ... ", end="")
    optimal_model_template = BTCModel(btc_historic_price=btc_historic_price_test, do_history=True)
    optimal_model_histories = []
    optimal_model_params = []
    for i in range(4):
        # Pass in the values of the best performing particle
        # (Note, could pass these to the constructor and call optimal_model.run()
        optimal_params = posterior_df.loc[i, priors.get_parameter_names()].to_dict()
        optimal_history = optimal_model_template(input_params_dict=optimal_params)  # run the model
        optimal_model_params.append(optimal_params)
        optimal_model_histories.append(optimal_history)

    # Plot performance, as above
    PLOT_MULTI = False
    if PLOT_MULTI: # Create figures with secondary y-axis (for the portfolio)
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}]],
        )
    else: # Otherwise just one figure (the best model)
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    row_col = [(1, 1), (1, 2), (2, 1), (2, 2)] if PLOT_MULTI else [(1,  1)]
    for i, (row, col) in enumerate(row_col):
        # BTC price (secondary axis)
        fig.add_trace(go.Scatter(x=df_test.index, y=df_test.Close,
                                 name='BTC price', line={'color': 'orange'}, showlegend=True if i == 1 else False),
                      row=row, col=col, secondary_y=True)

        # BTC balance
        fig.add_trace(go.Scatter(x=df_test.index, y=optimal_model_histories[i].to_df().BTC_Balance * 10000,
                                 mode='lines', name='btc balance (*1000)', line={'color': 'darkblue'},
                                 showlegend=True if i == 1 else False),
                      row=row, col=col, secondary_y=False)
        # Cash balance
        fig.add_trace(go.Scatter(x=df_test.index, y=optimal_model_histories[i].to_df().Cash_Balance,
                                 mode='lines', name='cash balance (*1000)', line={'color': 'lightblue'},
                                 showlegend=True if i == 1 else False),
                      row=row, col=col, secondary_y=False)
        # Portfolio balance
        fig.add_trace(go.Scatter(x=df_test.index, y=optimal_model_histories[i].to_df().Portfolio_Balance,
                                 mode='lines', name='portfolio value', line={'color': 'black', 'width': 2},
                                 showlegend=True if i == 1 else False),
                      row=row, col=col, secondary_y=False)

        # Add horizontal lines to show buy and sell behaviour
        for time, action in enumerate(optimal_model_histories[i].actions):
            if action == Action.BUY_BTC:
                fig.add_vline(x=df_test.index[time], line_width=2, line_dash="dash", line_color="green")
            elif action == Action.SELL_BTC:
                fig.add_vline(x=df_test.index[time], line_width=2, line_dash="dash", line_color="red")

        # Annotation of model parameters
        #fig.add_annotation(x=pd.Timestamp("2019"), y=np.max(optimal_model_histories[i].to_df().Portfolio_Balance),
        #                   text="\n".join( [f"{param}:{round(value, 3)}<br>" for param, value in optimal_model_params[i].items()]),
        #                   row=row, col=col )

    fig.update_layout(title_text="Portfolio performance")
    fig.update_xaxes(title_text="xaxis title")
    fig.update_yaxes(title_text="<b>optimal</b> performance", secondary_y=False)
    fig.update_yaxes(title_text="<b>BTC</b> price", secondary_y=True)
    plot(fig)
    print(" ... finished.")
