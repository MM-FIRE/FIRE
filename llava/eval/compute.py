#from llava.eval.capeval.bleu.bleu import Bleu
#from llava.eval.capeval.cider.cider import Cider
from capeval.bleu.bleu import Bleu
from capeval.cider.cider import Cider
# from nltk.tokenize import word_tokenize

scorer = Bleu()
predictions = {"IMG1": ["You correctly identified the color of the horse as brown, which is great! However, your answer about the horse being both baby and brown needs reconsideration. Consider the typical characteristics of horses in racing settings and how their size and appearance might indicate their age. Look closely at the body structure and the overall size of the horse compared to the cart it is pulling, which might give you more clues about its age."],
               "IMG2": ['You correctly identified the color of the horse as brown, which is great! However, your answer about the horse being both baby and brown needs reconsideration. Consider the typical characteristics of horses in racing settings and how their size and appearance might indicate their age. Look closely at the body structure and the overall size of the horse compared to the cart it is pulling, which might give you more clues about its age.']}
labels = {"IMG1": ["You correctly identified the color of the horse as brown, which is great! However, your answer about the horse being both baby and brown needs reconsideration. Consider the typical characteristics of horses in racing settings and how their size and appearance might indicate their age. Look closely at the body structure and the overall size of the horse compared to the cart it is pulling, which might give you more clues about its age."], "IMG2": ["You correctly identified the color of the horse as brown, which is great! However, your answer about the horse being both baby and brown needs reconsideration. Consider the typical characteristics of horses in racing settings and how their size and appearance might indicate their age. Look closely at the body structure and the overall size of the horse compared to the cart it is pulling, which might give you more clues about its age."]}


bleu = scorer.compute_score(labels, predictions)
bleu_4 = bleu[-1]
print("BLEU", bleu)

cider_scorer = Cider()
#labels = {"IMG1": ["hello"]}
#predictions = [{"image_id":"IMG1", "caption": ["hello"]}]
cider = cider_scorer.compute_score(labels, predictions)
print("Cider", cider)