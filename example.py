# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_example')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_example')
    _example = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_example', [dirname(__file__)])
        except ImportError:
            import _example
            return _example
        try:
            _mod = imp.load_module('_example', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _example = swig_import_helper()
    del swig_import_helper
else:
    import _example
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0


def new_intp():
    return _example.new_intp()
new_intp = _example.new_intp

def copy_intp(value):
    return _example.copy_intp(value)
copy_intp = _example.copy_intp

def delete_intp(obj):
    return _example.delete_intp(obj)
delete_intp = _example.delete_intp

def intp_assign(obj, value):
    return _example.intp_assign(obj, value)
intp_assign = _example.intp_assign

def intp_value(obj):
    return _example.intp_value(obj)
intp_value = _example.intp_value

def get_program(candidate, number):
    return _example.get_program(candidate, number)
get_program = _example.get_program

def neicun():
    return _example.neicun()
neicun = _example.neicun

def set_fitness(org, fitnessValue):
    return _example.set_fitness(org, fitnessValue)
set_fitness = _example.set_fitness

def get_fitness(org):
    return _example.get_fitness(org)
get_fitness = _example.get_fitness

def illegal(org):
    return _example.illegal(org)
illegal = _example.illegal

def action2Len(action2):
    return _example.action2Len(action2)
action2Len = _example.action2Len

def get_action2(action2, index):
    return _example.get_action2(action2, index)
get_action2 = _example.get_action2

def getPropertyfit(org):
    return _example.getPropertyfit(org)
getPropertyfit = _example.getPropertyfit

def getCheckedBySpin(org):
    return _example.getCheckedBySpin(org)
getCheckedBySpin = _example.getCheckedBySpin

def getroot(prog):
    return _example.getroot(prog)
getroot = _example.getroot

def getCandidate(candidate, i):
    return _example.getCandidate(candidate, i)
getCandidate = _example.getCandidate

def state_i(vector, i):
    return _example.state_i(vector, i)
state_i = _example.state_i

def judgeNULL(l):
    return _example.judgeNULL(l)
judgeNULL = _example.judgeNULL

def isNUll(p):
    return _example.isNUll(p)
isNUll = _example.isNUll

def findNode(root, prog, num):
    return _example.findNode(root, prog, num)
findNode = _example.findNode

def genVector(prog):
    return _example.genVector(prog)
genVector = _example.genVector

def copyProgram(prog):
    return _example.copyProgram(prog)
copyProgram = _example.copyProgram

def spin_(candidate):
    return _example.spin_(candidate)
spin_ = _example.spin_

def getLegalAction2(parent, nodeNum):
    return _example.getLegalAction2(parent, nodeNum)
getLegalAction2 = _example.getLegalAction2

def mutation1(parent, nodeNum, actionType):
    return _example.mutation1(parent, nodeNum, actionType)
mutation1 = _example.mutation1

def printAst(prog):
    return _example.printAst(prog)
printAst = _example.printAst

def set_coef(numofrequirements):
    return _example.set_coef(numofrequirements)
set_coef = _example.set_coef

def set_requirments(numofrequirements):
    return _example.set_requirments(numofrequirements)
set_requirments = _example.set_requirments

def fact(n):
    return _example.fact(n)
fact = _example.fact

def my_mod(x, y):
    return _example.my_mod(x, y)
my_mod = _example.my_mod

def get_time():
    return _example.get_time()
get_time = _example.get_time

def genInitTemplate(num):
    return _example.genInitTemplate(num)
genInitTemplate = _example.genInitTemplate

def genOrganism(templat):
    return _example.genOrganism(templat)
genOrganism = _example.genOrganism

def calculateFitness(prog, exp, numexp, coef):
    return _example.calculateFitness(prog, exp, numexp, coef)
calculateFitness = _example.calculateFitness

def freeAll(org, prog, t, c, e, type):
    return _example.freeAll(org, prog, t, c, e, type)
freeAll = _example.freeAll

def setAll(prog):
    return _example.setAll(prog)
setAll = _example.setAll

def initProg(requirements, numofrequirements, coef):
    return _example.initProg(requirements, numofrequirements, coef)
initProg = _example.initProg

def mutation_(candidate0, nodeNum, actType, requirements, numofrequirements, coef):
    return _example.mutation_(candidate0, nodeNum, actType, requirements, numofrequirements, coef)
mutation_ = _example.mutation_

def printprog(root, blank, prog):
    return _example.printprog(root, blank, prog)
printprog = _example.printprog

def getLength(action2):
    return _example.getLength(action2)
getLength = _example.getLength

def printAstint(prog):
    return _example.printAstint(prog)
printAstint = _example.printAstint
# This file is compatible with both classic and new-style classes.

cvar = _example.cvar

