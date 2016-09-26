import os
import json
from gensim.models import Word2Vec

print ' *', 'loading wv model'

modelFile = os.environ['HOME'] + "/models/" + "glove.6B.300d.txt"
#modelFile = os.environ['HOME'] + "/models/" + "glove.42B.300d.txt"
#modelFile = os.environ['HOME'] + "/models/" + "glove.twitter.27B.200d.txt"

model = Word2Vec.load_word2vec_format(modelFile, binary=False)
print ' *', 'model ready'

w1 = 'nostalgia'
w2 = 'memory'
print ' *', w1, w2, 'similarity:', model.similarity(w1, w2)

for w in ['nostalgia', 'blurred', 'figurative_art', 'erotic', 'voyeurism']:
    if w in model:
        print ' *', w, model.most_similar(positive=[w])


words = set([ "contemporary conceptualism", "appropriation", "contemporary participation", "colombian", "color photography", "american", "figurative art", "language", "abstract versus figurative art", "consumerism", "art that plays with scale", "architecture in art", "korean", "assemblage", "calarts", "collage", "1980s", "biomorphic", "collective history", "found objects", "grotesque", "cut/ripped", "decay", "united states", "flatness", "group of objects", "china", "chinese", "graffiti", "street art", "graffiti/street art", "color theory", "contemporary faux na\u00eff", "abstract sculpture", "art in art", "film/video", "singaporean", "cinematic", "brazil", "abstract", "brazilian", "'85 new wave", "city scenes", "drawing", "cultural commentary", "endurance art", "feminism", "bedrooms and bathrooms", "canadian", "columns and totems", "architecture's effects", "close-up", "1918 - 1939", "documentary photography", "black-and-white photography", "italian", "monochromatic", "gender", "globalization", "outdoor art", "mixed-media", "mexican", "mexico", "1990s", "ceramic", "animals", "artists' books", "1970s", "contemporary fact versus fiction", "art and technology", "installation art", "erased and obscured", "erotic", "contemporary grotesque", "etching/engraving", "abstract painting", "photoconceptualism", "bright/vivid", "abstract photography", "dark", "focus on materials", "contemporary traces of memory", "miniature and small-scale paintings", "conceptual", "photography", "japanese", "japan", "dutch", "contemporary vintage photography", "comic",
        "calligraphic", "belgium", "belgian", "contemporary surrealistic", "animation", "1960s", "collecting and modes of display", "cityscapes", "chance", "spain", "spanish", "black and white", "americana", "indian", "contemporary graphic realism", "conflict", "malaysian", "caricature / parody", "cross-cultural dialogue", "neo-conceptualism", "advertising and brands", "vietnamese", "australia and new zealand", "figurative painting", "central america", "el salvador", "food", "german-american", "germany", "puerto rican", "allover composition", "southern cone", "isolation", "sexual identity", "argentinean", "antiquity as subject", "contemporary archaeological", "human figure", "nude", "contemporary pop", "british", "indonesian", "anthropomorphism", "celebrity", "pakistani", "digital culture", "political", "violence", "social action", "contemporary diy", "narrative", "design", "architecture", "hard-edged", "minimalism", "flora", "chicano art", "crime", "color gradient", "contemporary color fields", "childhood", "suburbia", "blurred", "mexican american", "artist as ethnographer", "venezuelan", "humor", "figurative sculpture", "allegory", "focus on the social margins", "neo-concretism", "cuban", "myth/religion", "immersive", "modern", "pakistani-american", "angular", "costa rican", "abstract landscape", "body art", "performance art", "abject art", "light and space movement", "line, form and color", "classical mythology", "sculpture", "work on paper", "argentinian", "peruvian", "individual portrait", "automatism", "cuba", "engagement with mass media", "cubism", "emerging art"])

results = {}
unmatched = []

for w in words:
    x = w.replace(' ', '-')
    if x in model:
      results[w] = model.most_similar(positive=[x])
    else: 
      unmatched.append(w)

#print json.dumps(results, indent=True, ensure_ascii=False, encoding='utf-8')

print ' *', 'unmatched:', unmatched
print ' *', 'matched', len(results), 'of', len(words)
