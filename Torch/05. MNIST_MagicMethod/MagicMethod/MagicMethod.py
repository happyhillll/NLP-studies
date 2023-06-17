# magic method
# https://zzsza.github.io/development/2020/07/05/python-magic-method/

class Document:
    def __init__(self, x):
        self.sents = x
    def __add__(self, other):
        return Document(self.sents + other.sents)

    def __len__(self):
        return len(self.sents)


    def __getitem__(self, i):
        return self.sents[i]

doc_1 = Document(
    [
        "11", "22", "33"
    ]
)

doc_2 = Document(
    [
        "44", "55", "66"
    ]
)

print(len(doc_1))
new_doc = doc_1 + doc_2