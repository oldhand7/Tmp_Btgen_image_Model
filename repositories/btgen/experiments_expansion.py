from modules.expansion import BTGenExpansion

expansion = BTGenExpansion()

text = 'a handsome man'

for i in range(64):
    print(expansion(text, seed=i))
