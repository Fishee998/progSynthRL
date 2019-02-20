//
//  example.h
//  philoserpher
//
//  Created by 媛庄 on 2019/2/15.
//  Copyright © 2019年 媛庄. All rights reserved.
//

#ifndef example_h
#define example_h

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>



extern int numprivatevars;    //number of public variables of all programs
extern int numpublicvars;    //number of private variables of each program
extern int progid;
extern int maxconst;
extern int numprog;        //number of programs
extern int mutype;



typedef struct expression
{
    int type;    //0:CONST  1:VAR              //2:TIMES  3:PLUS  4:MINUS //5:DIV
    int index;    //0:value of CONST  1:index of VAR
    //VAR: -1:public variables    <=-2:me, other1,other2...
}exp_;

typedef struct semaphore
{
    int index;    //-1:mutex    -2:left -3:right
}sema;

typedef struct condition
{
    int type;
    //new -1:wi;    0:True;     1EQ;    2:NEQ

    exp_* exp1;
    exp_* exp2;
}cond;

typedef struct treenode
{
    int type;//0:IF  1:WHILE  2:SEQ 3:ASGN 4:WAIT 5:SIGNAL 6:think 7:eat
    int index;//ASSIGN left VAR index    -1:v[p]  -2:v[me]
    cond* cond1;
    struct treenode* treenode1;
    struct treenode* treenode2;
    struct treenode* next;
    struct treenode* parent;
    exp_* exp1;
    sema* sema1;
    int goodexamples;
    int badexamples;
    int pc;
    int depth;
    int height;
    int numofstatements;
    int fixed;
    int number;
}treenode;

typedef struct program
{
    treenode* root;
    int maxdepth;
    int progid;
    int maxconst;
    int numprivatevars;
    int numpublicvars;
    double fitness;
    int numofevent[7];
    double propertyfit[20];
    int checkedBySpin;
    int illegal;
}program;

typedef struct organism
{
    program** progs;
    double fitness;
    //int enterCS;
}organism;

typedef enum ExprType{Constant,Variable,Expr1,StepBoundExpr1,Expr2}ExprType;
typedef enum UnOp{Neg,Not,Future,Globally,Next}UnOp;
typedef enum BinOp{Imp,And,Or,Eq,Neq,Lt,Le,Gt,Ge,Add,Min,Mul,Div}BinOp;    //,Until,WeakUntil

typedef struct Expr
{
    ExprType type;
    UnOp uop;
    BinOp bop;

    int stepbound;
    char* name;
    int value;

    struct Expr* child;
    struct Expr* left;
    struct Expr* right;
}Expr;

typedef struct State{
    int numvar;
    int *varvalue;
    char**varname;
}State;

typedef struct trace
{
    int steplength;
    int numofprog;
    int numofvar;
    program** root;                //numofprog * program*
    //int** valueofvar;            //steplength * numofvar * int
    int** valueofsema;            //steplength * numofsema * int
    int* executeprogid;            //steplength * int
    treenode** executenode;        //steplength* treenode*
    int satisfied;
    State** states;                //(steplength + 1) * State*
}trace;

typedef struct mutationNode
{
    treenode* node;
    cond* cond;
    exp_* exp;
    int ranksum;
    int type;
}mutationNode;

typedef struct treeRank
{
    mutationNode** candidate;
    int numcandidate;
    int maxnumcandidate;
    int ranksum;
}treeRank;

int satisfyMutationReduction(treenode* t);
void mutationCond(cond* root,program* prog,int type);
//void mutationExp(exp* root,program* prog);
program* mutation(program* parent);

program** genNewCandidate(int numofcandidate,program** candidate,int numofmutation);
program** genNewCandidateWithCoefficient1(int numofcandidate,program** candidate,int numofmutation,double coef);
program** genNewCandidateWithCoefficient2(int numofcandidate,program** candidate,int numofmutation,double coef);

program** selectNewCandidate(int numofcandidate,program** candidate,int numofmutation, program** newcandidate, double* bord);
program** selectNewCandidateWithFitness(int numofcandidate,program** candidate,int numofmutation, program** newcandidate);

bool existNullCond(treenode* t);

void setbadexamples(trace** t,int num);
void setNext(treenode* root);
void setParent(treenode* root);

int checkEnterCS(trace* t);
extern double calculateFitness(organism* prog,Expr** exp,int numexp,double* coef);
void calculateFitness2(organism* prog,int type);

void initTraceGlobalVar(int steplength);
void freeTrace(trace* t);

//bool checkIfFixed(treenode* t);
bool statementsTooLong(treenode* t);
void printType(treenode* t);
int existCS(treenode* t);
exp_* createExp(int t,int ind);
sema* createSema(int ind);
cond* createCond(int t,exp_* e1,exp_* e2);
treenode* createTreenode(int t,int ind,cond* c,treenode* t1,treenode* t2,exp_* e);

int nextrand(int range);
void printprog(treenode* root,int blank,program* prog);
void orgToPml(organism* org,FILE* f);
void orgToRml(organism* org,FILE* f);
exp_* genexp(program* prog,int type);
cond* gencond(program* prog);
treenode* genprog(int depth,program* prog);
int* getPublicVarUsed(treenode* prog,int num);
treenode* genCS(int depth,program* prog);
void addCS(program* prog);
int equalExp(exp_* e1,exp_* e2);

void setNumOfStatements(treenode* root);
void setLinesTreenode(treenode* t,int depth);
int setFixed(treenode* t);
int getFixed(treenode* t);
void setAll(program* prog);

exp_* copyExp(exp_* e);
cond* copyCond(cond* c);
treenode* copyTreenode(treenode* t);
program* copyProgram(program* prog);

program* genProgram(program* templat,int progid);
organism* genOrganism(program* template);
program** genInitTemplate(int num);

void freeAll(organism* org,program* prog,treenode* t,cond* c,exp_* e,int type);
int compareTreenode(treenode* t1,treenode* t2);
treenode* example_e();

void addNode(treeRank* tr,treenode* node);
void searchNode(treenode* root,treeRank* tr,int type,int maxdepth);
mutationNode* chooseNode(treeRank* tr);

Expr* createExprConstant(int value);
Expr* createExprVar(char* name);
Expr* createExprExp1(UnOp op,Expr* child);
Expr* createExprStepb(UnOp op,Expr* child,int stepbound);
Expr* createExprExpr2(BinOp op,Expr* left,Expr* right);

Expr* generateStepExpr(char* buf, int leftindex, int rightindex);

int getStepBound(Expr* exp);

Expr** set_requirments(int numofrequirements);
int* genVector(program* prog);
treenode* findNode(treenode* root,program* prog, int num, treenode* result);
int* legalAction2(program* parent, int nodeNum);

extern int* genVector(program* prog);
extern void genVectorTreenode(treenode* node, int* id);
extern int getTreenodeId(treenode* node);
extern int getConditionId(cond* c);
extern int setTreenodeNum(treenode* root, program* prog, int number);
extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
extern program* copyProgram(program* prog);
extern program* initProg(Expr** requirements ,int numofrequirements);
extern int* genVector(program* prog);
extern program *mutation_(program* parent, int nodeNum, int actionNum, Expr** requirements);

#endif /* example_h */
