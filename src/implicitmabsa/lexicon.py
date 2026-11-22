"""Phrase lexicon for the ImplicitMABSA-Synth benchmark.

Every phrase below expresses an opinion about an aspect WITHOUT naming the aspect
(no "battery", "screen", "camera", ...).  This is what makes the aspect *implicit*:
a model must infer it from context (e.g. "Dead by lunch again" -> battery/negative).

Each list has 8 phrases.  The first 6 are used for train/val/IID-test; the last 2 are
held out and only appear in the OOD-test split (unseen phrasings).
"""

ASPECTS = ["battery", "screen", "camera", "price", "delivery", "packaging", "sound", "performance"]
SENTIMENTS = ["negative", "neutral", "positive"]  # label ids 0,1,2
N_SEEN = 6  # phrases [0:6] seen in training, [6:8] held out for OOD test

# --- aspect-specific opinion phrases: PHRASES[aspect][sentiment] -> 8 strings -------------------
PHRASES = {
    "battery": {
        "positive": [
            "Still going strong after two days without a plug",
            "Barely dipped after a full day of use",
            "Made it through the whole trip on a single top-up",
            "Forgot to plug it in last night and it's still fine",
            "Runs all week, I'm honestly shocked",
            "Easily outlasts my old one by hours",
            "Marathon session and it's still above half",
            "Doesn't even flinch when the day gets long",
        ],
        "negative": [
            "Dead by lunch again",
            "Gone from full to almost nothing in an hour",
            "Have to carry a cord everywhere now",
            "Shut off at 30% for no reason",
            "Needs a top-up every couple of hours",
            "Barely lasts through my commute",
            "My old one outlasted this by a mile",
            "Hot and empty within an hour",
        ],
        "neutral": [
            "Lasts about a day, nothing special",
            "Gets me to evening, as advertised",
            "Average endurance, no surprises",
            "Ends the day around 15%, so-so",
            "Needs plugging in nightly like every other one",
            "Same stamina as my last one",
            "Decent runtime for the size I suppose",
            "Somewhere in the middle for staying power",
        ],
        "topic": [
            "Plugged it in overnight as usual",
            "Checking the percentage again",
            "Top-up time before heading out",
            "Wall adapter is in the bag as always",
        ],
    },
    "screen": {
        "positive": [
            "Colors just pop, I keep staring at it",
            "Text looks crisp even in bright sun",
            "Watching movies on it is a treat",
            "So vivid, everything looks like a painting",
            "Readable outdoors at noon, love it",
            "Blacks are deep and everything looks silky",
            "Every video looks stunning now",
            "Scrolling feels smooth and gorgeous",
        ],
        "negative": [
            "Can't read a thing outside in daylight",
            "Looks washed out and yellowish",
            "Touch keeps missing my taps",
            "Flickers whenever I lower the brightness",
            "A crack appeared out of nowhere",
            "Everything looks grainy and dull",
            "Ghost images stay after I swipe",
            "Tilt it slightly and the colors invert",
        ],
        "neutral": [
            "Fine for emails, nothing to write home about",
            "Colors are okay, a bit average",
            "Bright enough indoors, dim outside",
            "Looks similar to my previous one",
            "Decent viewing, not stunning",
            "Gets the job done for scrolling",
            "Nothing wrong with how things look",
            "Average clarity for its class",
        ],
        "topic": [
            "Scrolled through my feed for a while",
            "Wiped the fingerprints off again",
            "Turned the brightness slider up",
            "Watched a couple of episodes tonight",
        ],
    },
    "camera": {
        "positive": [
            "Night shots look like daytime, unreal",
            "Every snap is frame-worthy",
            "Zoomed in and it's still razor sharp",
            "Portraits look professional, friends asked who shot them",
            "Captured the sunset exactly as I saw it",
            "Even my shaky-handed selfies turn out crisp",
            "Grandma's birthday came out gorgeous",
            "Low light? Not a problem for this thing",
        ],
        "negative": [
            "Everything I shoot comes out blurry",
            "Selfies look like potatoes",
            "Night shots are just noise",
            "Focus hunts forever then misses",
            "Colors look nothing like real life when I snap",
            "Blown-out highlights in every shot",
            "Autofocus gave up on my dog again",
            "Zoom turns everything into mush",
        ],
        "neutral": [
            "Snaps are okay in good light",
            "Decent shots, nothing to brag about",
            "Fine for social media posts",
            "Some shots turn out, some don't",
            "Comparable to my old one for snapshots",
            "Average results outdoors, meh indoors",
            "Serviceable for quick shots",
            "Does what a phone shooter should",
        ],
        "topic": [
            "Took a few shots at the park today",
            "Framing the skyline for my story",
            "Tried the portrait mode toggle",
            "Snapped my lunch again",
        ],
    },
    "price": {
        "positive": [
            "Worth every penny and then some",
            "Got so much for so little",
            "My wallet is still smiling",
            "A steal, honestly",
            "Didn't have to skip lunch to buy it",
            "Bargain of the year for me",
            "Paid less than my friend and got more",
            "Couldn't believe it was on sale for that",
        ],
        "negative": [
            "Ouch, my wallet is crying",
            "Way more than it's worth",
            "Highway robbery for what you get",
            "Could have bought two rivals for this",
            "Fancy label, empty value",
            "Felt scammed at checkout",
            "Overpaying for a logo, honestly",
            "Not worth what they're asking",
        ],
        "neutral": [
            "About what I expected to pay",
            "Fair for what it is, I guess",
            "In line with similar ones out there",
            "Neither a bargain nor a rip-off",
            "Average tag for the category",
            "Middle of the road on value",
            "Same total at checkout as the others out there",
            "Reasonable, sort of",
        ],
        "topic": [
            "Checked the receipt again",
            "Compared a few listings before ordering",
            "Added to cart and stared at the total",
            "Waiting for the sale banner",
        ],
    },
    "delivery": {
        "positive": [
            "Showed up a day early, nice surprise",
            "Landed on my doorstep before I finished the order confirmation",
            "Right on time, even in the rain",
            "Was at my door within hours",
            "Got here well ahead of the promised date",
            "Rider called ahead and handed it over politely",
            "Already on my porch before my morning coffee",
            "Zero waiting, that was quick",
        ],
        "negative": [
            "Still waiting after two weeks",
            "Showed up a week late with no apology",
            "Status page hasn't budged in days",
            "Left outside in the rain",
            "Nobody came to my door, marked as attempted",
            "Took forever to reach me",
            "Wrong address, of course",
            "Promised Tuesday, got it next month",
        ],
        "neutral": [
            "Came on the date they said",
            "Arrived when expected, nothing more",
            "Took the usual few days",
            "Standard wait, standard result",
            "Reached me around the middle of the estimate window",
            "No complaints and no praise on the wait",
            "Turned up eventually, as they do",
            "Normal timeline, I'd say",
        ],
        "topic": [
            "Watching the doorstep every hour",
            "Refreshing the order status again",
            "Someone rang the bell just now",
            "Estimated date is tomorrow",
        ],
    },
    "packaging": {
        "positive": [
            "Opening it felt like a gift, so neatly done",
            "Everything snug and cushioned inside",
            "Not a single scratch, so carefully protected",
            "Even the tape was tidy, lovely touch",
            "Arrived pristine, someone cared",
            "Eco-friendly and beautifully presented",
            "Like a present, each item nested in tissue",
            "Protected like a museum piece",
        ],
        "negative": [
            "Arrived crushed with the corner caved in",
            "Torn open and taped back up sloppily",
            "Rattled loose inside like a bag of nuts",
            "Soaked through with nothing to protect it",
            "Way too much plastic and it still got dented",
            "Ripped everywhere before I even touched the contents",
            "Looked like it was kicked down the stairs",
            "Just tossed in an envelope, seriously",
        ],
        "neutral": [
            "Standard cardboard, does the job",
            "Plain and functional outside",
            "Nothing fancy in how it was put together",
            "Adequate protection, nothing special",
            "Came in the usual brown outer",
            "Looked ordinary but intact",
            "Neither lovely nor terrible on presentation",
            "Same brown outer as everyone else",
        ],
        "topic": [
            "Cutting the tape with scissors",
            "Flattening cardboard for recycling",
            "Peeling off the outer layer now",
            "Opened it on the kitchen table",
        ],
    },
    "sound": {
        "positive": [
            "Every song feels like a live concert",
            "Crisp highs and a real punch in the low end",
            "Podcasts feel like the host is sitting beside me",
            "Could hear tiny details I never noticed before",
            "Rich and full, my neighbours asked what I was playing",
            "Vocals are silky, I got goosebumps",
            "Movie nights are cinematic now",
            "Fills the whole room without any strain",
        ],
        "negative": [
            "Everything is tinny and hollow",
            "Crackles like an old radio",
            "Voices come out muffled and distant",
            "Buzzing every time the beat drops",
            "Can't hear a thing on calls",
            "Distorts when I turn it up",
            "Left channel just cuts out",
            "Like listening from inside a tin can",
        ],
        "neutral": [
            "Fine for casual listening",
            "Okay for calls and background tunes",
            "Nothing that will wow a purist",
            "Passable for the office",
            "Middle-of-the-road clarity",
            "Not bad, not great when you crank it",
            "Average for something this size",
            "Clear enough for videos, that's all",
        ],
        "topic": [
            "Put my earbuds in on the train",
            "Plugged the jack in again",
            "Queued up my playlist",
            "Testing the Bluetooth pairing",
        ],
    },
    "performance": {
        "positive": [
            "Apps open before I lift my finger",
            "Multitasking feels effortless",
            "Games run buttery smooth",
            "Never stutters, no matter how many tabs",
            "Boots up in seconds",
            "Editing videos without a hiccup",
            "Handled everything I threw at it",
            "Instant response to every tap",
        ],
        "negative": [
            "Keeps freezing right when I need it",
            "Takes ages to open anything",
            "Stutters every time I switch apps",
            "Crashed twice while I was typing",
            "Gets hot and crawls",
            "Waiting on the spinner all day",
            "Games turn into slideshows",
            "Restarts itself randomly",
        ],
        "neutral": [
            "Handles daily tasks, struggles with heavy ones",
            "Okay for browsing, that's about it",
            "Decent for the basics",
            "Occasional hiccup but manageable",
            "Similar responsiveness to my last one",
            "Gets things done in its own time",
            "Manageable if I close a few apps",
            "Adequate for what I do",
        ],
        "topic": [
            "Opened a few tabs to test it",
            "Installing my usual apps tonight",
            "Restarted it and tried again",
            "Ran the benchmark app once",
        ],
    },
}

# --- aspect-agnostic sentiment phrases (no aspect information at all) --------------------------
GENERIC_SENT = {
    "positive": ["Absolutely love it!", "Totally worth it", "Exceeded my expectations", "Best purchase this year",
                 "So happy right now", "Highly recommend to everyone", "Beyond thrilled with this one", "Zero regrets"],
    "negative": ["Not what I expected!", "Really disappointed", "Total letdown", "Would not recommend",
                 "Regret buying this", "Very unhappy with this one", "What a waste", "Never again, seriously"],
    "neutral": ["It's okay I guess", "Neither good nor bad", "Meh", "Does what it says",
                "About average", "Not sure how I feel about it", "Fine, I suppose", "Nothing special either way"],
}
# --- uninformative filler (no aspect, no sentiment) ---------------------------------------------
FILLER = ["Just got mine", "Posting an update", "Here's my take", "Thoughts after a week",
          "Finally sharing my review", "Here we go", "Update time", "Got it yesterday"]

OPENERS_SEEN = ["", "", "", "Honestly, ", "Update: ", "Ok so ", "Well, ", "Day 3: "]
OPENERS_OOD = ["Quick note: ", "FWIW ", "Btw ", "Real talk: "]
CLOSERS_SEEN = ["", "", "", "", " #review", " #gadgets", " #unfiltered", " lol", "..."]
CLOSERS_OOD = [" #techtalk", " tbh", " #verdict", " smh"]
EMOJI = {"positive": ["😍", "👍", "🔥", "😊"], "negative": ["😡", "👎", "😞", "😤"], "neutral": ["😐", "🤷"]}

# Words that would make an aspect EXPLICIT.  The test-suite asserts none appear in any phrase.
EXPLICIT_TERMS = {
    "battery": ["battery", "charge", "charging", "charger"],
    "screen": ["screen", "display", "monitor", "panel"],
    "camera": ["camera", "photo", "picture", "lens"],
    "price": ["price", "cost", "expensive", "cheap", "pricey", "afford", "money", "dollar", "$"],
    "delivery": ["deliver", "shipping", "shipped", "courier", "dispatch"],
    "packaging": ["package", "packaging", "packed", "box", "carton", "parcel", "wrap"],
    "sound": ["sound", "audio", "speaker", "volume", "bass", "treble", "music"],
    "performance": ["performance", "speed", "fast", "slow", "lag"],
}
