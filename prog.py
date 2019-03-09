import example
import astEncoder

numofrequirements = 12
def some():
    requirements = example.set_requirments(numofrequirements)
    return requirements

def initProg():
    requirements = some()
    # 100 candidates
    candidate = example.initProg(requirements, numofrequirements)

    return candidate

def actionLegal():
    # state and treenodeNum
    ast = astEncoder.getAstDict()
    state, astActNode = astEncoder.astEncoder(ast)
    requirements = some()
    # nodeNum: selected node  number  nodth: selected node geographical location
    nodeNum, nodth = astEncoder.get_action1(astActNode, ast, actionSet)
    # actionType: action2
    actionType = astEncoder.getAction2(nodth)
    print(nodeNum, actionType)

def mutation(candidate, action):
    requirements = some()
    candidate1 = example.mutation_new(candidate, action, requirements, numofrequirements)
    #root = example.getroot(candidate1)
    #example.printprog(root, 0, candidate1)
    return candidate1
