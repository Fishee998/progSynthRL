%module example
%include <cpointer.i>
%pointer_functions(int, intp);
%{
#include "example.h"
%}

%inline %{
/* array[i] */
program* get_program(program** candidate,int number)
{
    return candidate[number];
}

program* neicun()
{
    program* newcandidate = (program*)malloc(sizeof(program));
    return newcandidate;
}

program* set_fitness(program* org, double fitnessValue)
{
    org->fitness = fitnessValue;
    return org;
}

double get_fitness(program* org)
{
    return org->fitness;
}

int illegal(program* org)
{
    return org->illegal;
}
int action2Len(int* action2)
{
    return sizeof(action2);
}
int get_action2(int* action2, int index)
{
    return action2[index];
}
double getPropertyfit(program* org)
{
    return org->propertyfit[0];
}
int getCheckedBySpin(program* org)
{
    return org->checkedBySpin;
}
treenode* getroot(program* prog)
{
    return prog->root;
}
program* getCandidate(program** candidate, int i)
{
    return candidate[i];
}

int state_i(int* vector, int i)
{
    return vector[i];
}
int intProg(int i)
{
    extern int *gp_progint;
    return gp_progint[i];
}
int judgeNULL(treenode* l)
{
    if(l == NULL)
        return 1;
    else
        return 0;
}
int isNUll(program* p)
{
    if(p == NULL)
        return 0;
    else
        return 1;
}
%}
treenode* findNode(treenode* root,program* prog, int num);
extern int* genVector(program* prog);
extern program* copyProgram(program* prog);
extern int spin_(program* candidate);
extern int* getLegalAction2(program* parent, int nodeNum);
extern program* mutation1(program* parent, int nodeNum, int actionType);
extern void printAst(program* prog);
extern double* set_coef(int numofrequirements);
extern Expr** set_requirments(int numofrequirements);
extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
extern program** genInitTemplate(int num);
extern organism* genOrganism(program* templat);
extern double calculateFitness(organism* prog,Expr** exp,int numexp,double* coef);
extern void freeAll(organism* org,program* prog,treenode* t,cond* c,exp_* e,int type);
extern void setAll(program* prog);
extern program** initProg(Expr** requirements ,int numofrequirements,double* coef);
extern program* mutation_(program* candidate0, int nodeNum, int actType,Expr** requirements ,int numofrequirements,double* coef);
extern void printprog(treenode* root,int blank,program* prog);
extern int getLength(int* action2);
extern void printAstint(program* prog);
