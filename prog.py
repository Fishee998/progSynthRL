import example
import astEncoder

numofrequirements = 7
def some():
    requirements = example.set_requirments(numofrequirements)
    coef = example.set_coef(numofrequirements)
    actionSet = astEncoder.setActSet()
    return requirements, coef, actionSet

def initProg():
    requirements, coef, actionSet = some()
    # 100 candidates
    candidates = example.initProg(requirements, numofrequirements, coef)
    return candidates

def actionLegal():
    # state and treenodeNum
    ast = astEncoder.getAstDict()
    state, astActNode = astEncoder.astEncoder(ast)
    requirements, coef, actionSet = some()
    # nodeNum: selected node  number  nodth: selected node geographical location
    nodeNum, nodth = astEncoder.get_action1(astActNode, ast, actionSet)
    # actionType: action2
    actionType = astEncoder.getAction2(nodth)
    print(nodeNum, actionType)

def mutation(candidate, nodeNum, actionType):
    requirements, coef, actionSet = some()
    '''
    root = example.getroot(candidate)
    example.printprog(root, 0, candidate)
    print("action1: ", nodeNum)
    print("action2: ", actionType)
    '''
    candidate1 = example.mutation_(candidate, nodeNum, actionType, requirements, numofrequirements, coef)
    return candidate1
