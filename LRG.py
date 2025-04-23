class LRG(object):
    def __init__(self, N, V, P, S, H, end='.'):
        self.nonterm = set(N)
        self.term = set(V)
        self.term.add('ε')
        self.prod = {}
        for lhs, rhs in P.items():
            self.prod[lhs] = set()
            for part in rhs.split('|'):
                if len(part) > 0:
                    self.prod[lhs].add(part)
        self.S = S
        self.H = H
        self.end = end
        self.term.add(self.end)
        self.reduction_table = {}
        self.update_reduction_table()

    def update_reduction_table(self):
        # ресет таблицы
        self.reduction_table = {}
        # перебираем все комбинации нетерминал+терминал
        # для каждого нетерминала в мн-ве нетерминалов без S, но с H
        for n in self.nonterm.difference(self.S).union(self.H):
            # для каждого терминала в мн-ве терминалов без пустой цепочки
            for t in self.term.difference('ε'):
                state = str(n + t)
                # задаем значение по умолчанию
                self.reduction_table[state] = '-'
                # для каждого A=lhs и alpha=rhs в мн-ве правил
                for lhs, rhs in self.prod.items():
                    # для каждой части правила rhs
                    for part in rhs:
                        if state == part:
                            self.reduction_table[state] = lhs
                            continue
                        if n == self.H:
                            if t == part:
                                self.reduction_table[state] = lhs

    def produces(self, sentence):
        n = self.H
        for t in sentence:
            value = self.reduction_table[str(n + t)]
            if value == '-':
                return False
            n = value
        return True


lrg = LRG(['S', 'A', 'B'], ['a', 'b'], {
    'S': 'A.',
    'A': 'Ab|Bb|b',
    'B': 'Aa'
}, 'S', 'H', end='.')

print('Параметры регулярной левосторонней грамматики lrg:')
print('N: ', lrg.nonterm)
print('V: ', lrg.term)
print('P: ', lrg.prod)
print('S: ', lrg.S)
print('Начальное состояние: ', lrg.H)
print('Символ конца цепочки: ', lrg.end)
print('Таблица сверток: ', lrg.reduction_table)
print()
while True:
    print('Введите цепочку или введите exit для выхода...')
    response = input()
    if response == 'exit':
        break
    if lrg.produces(response):
        print('цепочка ' + response + ' принадлежит грамматике')
    else:
        print('цепочка ' + response + ' не принадлежит грамматике')
