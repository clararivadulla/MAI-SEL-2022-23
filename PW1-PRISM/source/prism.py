import pandas as pd


def prism(x):
    P = {}  # Prism ← ∅
    C = x['class'].unique()  # unique classes

    for C_i in C:  # for each class Ci do

        E = x[x['class'] == C_i]

        P[C_i] = {}

        i = 1

        while not E.empty:  # while E ≠ ∅

            perfect = False

            available_attr = list(E.columns)
            available_attr.remove('class')
            if 'index' in available_attr:
                available_attr.remove('index')

            rule_attr_vals = []

            while not perfect and available_attr:

                max_val_tmp = 0
                max_attr_vals = (None, None)

                for attr in available_attr:

                    values = list(E[attr].unique())

                    for val in values:

                        temp_rule_attr_vals = rule_attr_vals.copy()
                        temp_rule_attr_vals.append((attr, val))

                        aux_E = E.copy()
                        for attr, val in temp_rule_attr_vals:
                            aux_E = aux_E.loc[aux_E[attr] == val]
                        positive = len(aux_E)

                        aux_x = x.copy()
                        for attr, val in temp_rule_attr_vals:
                            aux_x = aux_x.loc[aux_x[attr] == val]
                        total = len(aux_x)

                        if total > 0:
                            p_t = positive / total
                        else:
                            p_t = 0

                        if p_t > max_val_tmp:
                            max_val_tmp = p_t
                            max_attr_vals = (attr, val)

                rule_attr_vals.append(max_attr_vals)
                available_attr.remove(max_attr_vals[0])

                if max_val_tmp == 1:
                    perfect = True

            aux_E = E.copy()
            for attr, val in rule_attr_vals:
                aux_E = aux_E.loc[aux_E[attr] == val]

            E = pd.merge(E, aux_E, how='outer', indicator=True)
            E = E[E['_merge'] == 'left_only'].drop('_merge', axis=1)

            P[C_i][i] = rule_attr_vals
            i += 1

    return P


def classify(P, x):
    predictions = []
    for index, row in x.iterrows():
        predictions.append(predict_class(row, P))
    return predictions


def satisfies_rule(xi, rule):
    for attr, val in rule:
        if xi[attr] != val:
            return False
    return True


def predict_class(xi, P):
    for C_i in P:
        for rule in P[C_i]:
            rule_attr_vals = P[C_i][rule]
            satisfy = satisfies_rule(xi, rule_attr_vals)
            if (satisfy):
                return C_i
    return None
