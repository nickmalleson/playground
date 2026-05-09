"""Data model and seed data for the book recommender."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal


GenreKey = Literal["cli-fi", "cyber", "space", "fantasy", "historical", "literary"]
MarkLike = Literal["love", "meh"]
MarkLabel = Literal["loved", "passed", "read"]

VALID_GENRES = {"cli-fi", "cyber", "space", "fantasy", "historical", "literary"}


@dataclass
class Book:
    id: str
    title: str
    author: str
    genre: str
    why: str
    is_fresh: bool = False


@dataclass
class Mark:
    read: bool = False
    like: Optional[str] = None  # "love" | "meh" | None

    def label(self) -> Optional[str]:
        if self.like == "love":
            return "loved"
        if self.like == "meh":
            return "passed"
        if self.read:
            return "read"
        return None

    def is_marked(self) -> bool:
        return self.read or self.like is not None


@dataclass
class HistoryEntry:
    title: str
    author: str
    genre: str
    mark: str  # "loved" | "passed" | "read"
    replaced_at: str  # ISO 8601


GENRES = [
    {"key": "cli-fi",     "label": "Climate dystopia",            "sub": "For when you want Bacigalupi, Atwood or McCarthy"},
    {"key": "cyber",      "label": "Cyberpunk and tech-noir",     "sub": "For when you want Gibson"},
    {"key": "space",      "label": "Space opera and hard sci-fi", "sub": "For when you want Hyperion, Three Body or Dune"},
    {"key": "fantasy",    "label": "Epic fantasy",                "sub": "For when you want Abercrombie, GRRM or Lynch"},
    {"key": "historical", "label": "Historical fiction",          "sub": "For when you want Cornwell or Harris"},
    {"key": "literary",   "label": "Literary speculative",        "sub": "For when you want Le Guin or Mitchell"},
]


ORIGINAL_FAVOURITES = [
    "Margaret Atwood — Oryx and Crake",
    "William Gibson — The Sprawl Trilogy and most other Gibson",
    "Cormac McCarthy — The Road",
    "Tim Winton — Juice",
    "Paolo Bacigalupi — all books, especially The Water Knife and The Windup Girl",
    "George R. R. Martin — A Game of Thrones",
    "Patrick Rothfuss — The Kingkiller Chronicle",
    "Joe Abercrombie — all books, especially The First Law",
    "Cixin Liu — The Three-Body Problem and the rest of the trilogy",
    "Dan Simmons — the Hyperion Cantos",
    "Robert Harris — all books",
    "Bernard Cornwell — especially the Saxon Stories (Uhtred and Alfred the Great)",
    "Ursula K. Le Guin — sci-fi and fantasy",
    "David Mitchell — Cloud Atlas",
    "Frank Herbert — Dune (and posthumous continuations)",
    "Scott Lynch — Gentleman Bastard / Locke Lamora",
]


SEED_BOOKS = [
    # Climate dystopia
    Book("parable-of-the-sower", "Parable of the Sower", "Octavia E. Butler", "cli-fi",
         "Walled communities, climate collapse, a teenage prophet on the road. The bone-deep dread of The Road meets the political thrust of Bacigalupi — and it was written in 1993."),
    Book("ministry-future", "The Ministry for the Future", "Kim Stanley Robinson", "cli-fi",
         "A polyphonic climate novel that opens with a heatwave scene as devastating as anything in The Water Knife, then pivots into oddly hopeful policy wonkery."),
    Book("american-war", "American War", "Omar El Akkad", "cli-fi",
         "A second US civil war in the 2070s, fought over the last fossil fuels. Spare, devastating prose — distinctly McCarthy-coded, with a slow-burn protagonist arc."),
    Book("gold-fame-citrus", "Gold Fame Citrus", "Claire Vaye Watkins", "cli-fi",
         "California's water is gone and a dune sea is swallowing the Southwest. Lyrical and hallucinatory — basically a literary sibling to The Water Knife."),

    # Cyberpunk
    Book("snow-crash", "Snow Crash", "Neal Stephenson", "cyber",
         "Cyberpunk's other foundational text — funnier and faster than Gibson. Samurai pizza couriers, a metaverse linguistic virus, and the best opening 50 pages in the genre."),
    Book("altered-carbon", "Altered Carbon", "Richard Morgan", "cyber",
         "Hard-boiled detective noir plus downloadable consciousness. The Sprawl trilogy's 2000s grandchild — grittier, pulpier, and very confident."),
    Book("quantum-thief", "The Quantum Thief", "Hannu Rajaniemi", "cyber",
         "A post-cyberpunk caper across a transhuman solar system. Dense, dazzling, throws you in the deep end exactly the way Neuromancer did."),
    Book("blindsight", "Blindsight", "Peter Watts", "cyber",
         "First-contact horror with transhumans and predator-vampires interrogating consciousness itself. Brutally smart — Hyperion fans tend to love it too."),

    # Space opera & hard SF
    Book("memory-called-empire", "A Memory Called Empire", "Arkady Martine", "space",
         "A small-station ambassador navigates the seductive horror of a vast empire. Reads like Hyperion crossed with Le Guin's anthropological eye. The sequel is just as good."),
    Book("player-of-games", "The Player of Games", "Iain M. Banks", "space",
         "The friendliest doorway into the Culture. Scope, civilisational politics, and moral weight that should hit your Dune and Hyperion nerve. Banks is the missing author from your list."),
    Book("children-of-time", "Children of Time", "Adrian Tchaikovsky", "space",
         "Uplifted spiders evolve over millennia while a battered generation ship limps toward them. Big-idea SF in Cixin Liu's lineage but with more empathy."),
    Book("house-of-suns", "House of Suns", "Alastair Reynolds", "space",
         "A six-million-year galactic timescale, with clones of a single woman trading memories at meet-ups every 200,000 years. Hyperion-grade scope and melancholy."),

    # Epic fantasy
    Book("gardens-of-the-moon", "Gardens of the Moon (Malazan)", "Steven Erikson", "fantasy",
         "An army-scale, mythologically dense fantasy spread across continents and millennia. Steeper learning curve than First Law but the payoff over ten books is enormous."),
    Book("black-company", "The Black Company", "Glen Cook", "fantasy",
         "Grimdark before grimdark was a word. Mercenary-company POV, blood-and-mud morality — the obvious ancestor of Abercrombie's First Law."),
    Book("assassins-apprentice", "Assassin's Apprentice", "Robin Hobb", "fantasy",
         "Slow-burn, ferocious character work across a 16-book braided saga. The author Rothfuss readers most often graduate to — and she actually finishes her series."),
    Book("poppy-war", "The Poppy War", "R. F. Kuang", "fantasy",
         "Military academy to genocide-scale war, drawn from 20th-century Chinese history. Brutal, modern, GRRM-territory stakes — not for the faint-hearted."),

    # Historical fiction
    Book("wolf-hall", "Wolf Hall", "Hilary Mantel", "historical",
         "Thomas Cromwell's rise in Henry VIII's court. Prose good enough to be illegal, plus all the political cunning of GRRM but in actual history. The trilogy is one of the great achievements."),
    Book("shogun", "Shōgun", "James Clavell", "historical",
         "An Englishman shipwrecked into Sengoku-era Japan. Massive, addictive, written with the same propulsive instinct as Cornwell's Saxon books."),
    Book("master-and-commander", "Master and Commander", "Patrick O'Brian", "historical",
         "Napoleonic naval fiction across 20 books. The more literary cousin of Cornwell's Sharpe — quieter, funnier, profoundly companionable once you tune in to the rhythm."),
    Book("hhhh", "HHhH", "Laurent Binet", "historical",
         "The 1942 plot to assassinate Reinhard Heydrich, told as a meta-narrative about the difficulty of telling it. Slim, riveting — Robert Harris would adore this one."),

    # Literary speculative
    Book("piranesi", "Piranesi", "Susanna Clarke", "literary",
         "A man lives alone in an infinite labyrinth of statues and tides. Short, strange, perfect — Le Guin would have loved it."),
    Book("never-let-me-go", "Never Let Me Go", "Kazuo Ishiguro", "literary",
         "Quiet dystopia disguised as a boarding-school memoir. Devastating in the same way Cloud Atlas is devastating, but more concentrated."),
    Book("jonathan-strange", "Jonathan Strange & Mr Norrell", "Susanna Clarke", "literary",
         "Two magicians revive English magic during the Napoleonic Wars. Footnoted, funny, immense — the texture of Le Guin with the wit of an Austen novel."),
    Book("memory-police", "The Memory Police", "Yoko Ogawa", "literary",
         "On a small island, things vanish from memory one by one — birds, ribbons, novels, body parts. Ishiguro-quiet, Atwood-political, hauntingly beautiful."),
]


def genre_label(key: str) -> str:
    for g in GENRES:
        if g["key"] == key:
            return g["label"]
    return key


def genre_sub(key: str) -> str:
    for g in GENRES:
        if g["key"] == key:
            return g["sub"]
    return ""
