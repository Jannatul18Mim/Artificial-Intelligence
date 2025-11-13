import math

def entropy(y):
    from collections import Counter
    n = len(y)
    c = Counter(y)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def majority(y):
    from collections import Counter
    return Counter(y).most_common(1)[0][0]

def info_gain(X, y, feat):
    n = len(y)
    vals = {x[feat] for x in X}
    e = entropy(y)
    s = 0
    for v in vals:
        idx = [i for i, x in enumerate(X) if x[feat] == v]
        if not idx:
            continue
        py = [y[i] for i in idx]
        s += len(idx)/n * entropy(py)
    return e - s

def build(X, y, feats):
    if len(set(y)) == 1:
        return ('leaf', y[0])
    if not feats:
        return ('leaf', majority(y))
    gains = [(info_gain(X, y, f), f) for f in feats]
    gains.sort(reverse=True)
    best = gains[0][1]
    vals = {x[best] for x in X}
    node = ('node', best, {})
    for v in vals:
        idx = [i for i, x in enumerate(X) if x[best] == v]
        if not idx:
            node[2][v] = ('leaf', majority(y))
        else:
            Xv = [X[i][:best] + X[i][best+1:] for i in idx]
            yv = [y[i] for i in idx]
            fv = [f for f in feats if f != best]
            fv = [f - 1 if f > best else f for f in fv]
            child = build(Xv, yv, fv)
            node[2][v] = child
    return node

def predict(tree, x):
    t = tree
    while t[0] == 'node':
        f = t[1]
        v = x[f]
        if v not in t[2]:
            return None
        x = x[:f] + x[f+1:]
        t = t[2][v]
    return t[1]

def print_tree(tree, feature_names, depth=0):
    prefix = "  " * depth
    if tree[0] == 'leaf':
        print(f"{prefix}Leaf: {tree[1]}")
    else:
        feat = tree[1]
        for val, subtree in tree[2].items():
            print(f"{prefix}If {feature_names[feat]} == {val}:")
            print_tree(subtree, feature_names[:feat] + feature_names[feat+1:], depth + 1)

if __name__ == "__main__":
    dataset = [
        ['Sunny', 'Hot', 'High', False, 'No'],
        ['Sunny', 'Hot', 'High', True, 'No'],
        ['Overcast', 'Hot', 'High', False, 'Yes'],
        ['Rain', 'Mild', 'High', False, 'Yes'],
        ['Rain', 'Cool', 'Normal', False, 'Yes'],
        ['Rain', 'Cool', 'Normal', True, 'No'],
        ['Overcast', 'Cool', 'Normal', True, 'Yes'],
        ['Sunny', 'Mild', 'High', False, 'No'],
        ['Sunny', 'Cool', 'Normal', False, 'Yes'],
        ['Rain', 'Mild', 'Normal', False, 'Yes'],
        ['Sunny', 'Mild', 'Normal', True, 'Yes'],
        ['Overcast', 'Mild', 'High', True, 'Yes'],
        ['Overcast', 'Hot', 'Normal', False, 'Yes'],
        ['Rain', 'Mild', 'High', True, 'No']
    ]

    # Split features and labels
    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    # Encode categorical and boolean features to integers
    m = {}
    idc = 0
    Xenc = []
    for row in X:
        r = []
        for v in row:
            if v not in m:
                m[v] = idc
                idc += 1
            r.append(m[v])
        Xenc.append(r)

    feature_names = ["Outlook", "Temperature", "Humidity", "Wind"]

    tree = build(Xenc, y, list(range(len(Xenc[0]))))

    print("Decision Tree Structure:")
    print_tree(tree, feature_names)

    # Example prediction
    x = ['Sunny', 'Cool', 'High', True]
    xenc = [m[v] for v in x]
    print("\nPrediction for", x, ":", predict(tree, xenc))
