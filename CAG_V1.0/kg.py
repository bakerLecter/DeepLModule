# from utils import Prepare_DataSet
from kg import Kg_Inject
import logging

text = '一开始读了很多次都没能读完，后来自己经历了一些打击，心境慢下来，才又接着读完，后面看得就很入[OnO]味，能感受到故事的忧伤，作者功力其实很深，看似白描的手法，将很多激烈的场景和感情轻描淡写地写于纸上，但其中境况，只要稍微一品，就能觉出。公主和宦官的爱情，听上去很猎艳，但这个故事非常凄美，世界上哪有那么多偶像剧一样的人生，就算贵为公主，也必须活在皇家的桎梏中，履行公主应尽的职责，不能放手去爱，甚至妥协了自己的爱情。故事最惊心动魄的情节在我看来是二人的诀别那夜，怀吉在公主睡着之后离开，仍然是白描的写法，说他看见公主的睡脸，这是此生见她的最后一面。看到这段时心如重锤，短短数字已令我泣不成声，也许是因为现在的我已有深爱的人，知道和挚爱分离是多么不能想象的一件事，所以才对这种禁忌的爱情感触颇深吧。好文，太虐'

KGI = Kg_Inject.KnowledgeInject(KGPath = "./datasets/CnDbpedia.spo", MaxEntitiesSelect=2)
# print(len(KGI.kg))

sent = text.strip().split('\t')
words = sent[0]

test = KGI.KnowledgeInjector(words,1)
processed = []
for k in test:
    t = k.replace('[OnO]','')
    processed.append(t)

print(processed)