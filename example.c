//
//  example.c
//  philoserpher
//
//  Created by 媛庄 on 2019/2/15.
//  Copyright © 2019年 媛庄. All rights reserved.
//

#include "example.h"
#include <stdbool.h>
#include <stdio.h>


int numprog = 3;
double genseq = 0.4;

/*bool checkIfFixed(treenode* t)
 {
 if(t == NULL)
 return false;
 if(t->type == 4 || t->type == 5)
 return false;
 
 if(t->type == 0)
 if(getFixed(t->treenode1) == 1 || getFixed(t->treenode2) == 1)
 return true;
 return (checkIfFixed(t->treenode1) || checkIfFixed(t->treenode2));
 }*/

// init.c
int equalExp(exp_* e1,exp_* e2)
{
    if(e1 == NULL || e2 == NULL)
        return 0;
    if(e1->type == e1->type && e1->index == e2->index)
        return 1;
    return 0;
}

exp_* createExp(int t,int ind)
{
    exp_* p = (exp_*)malloc(sizeof(exp_));
    p->type = t;
    p->index = ind;
    return p;
}

sema* createSema(int ind)
{
    sema* s = (sema*)malloc(sizeof(sema));
    s->index = ind;
    return s;
}

cond* createCond(int t,exp_* e1,exp_* e2)
{
    cond* p = (cond*)malloc(sizeof(cond));
    p->type = t;
    p->exp1 = e1;
    p->exp2 = e2;
    return p;
}

treenode* createTreenode(int t,int ind,cond* c,treenode* t1,treenode* t2,exp_* e)   //????? 数组初始化
{//printf("createtreenode type:%d",t);
    treenode* p = (treenode*)malloc(sizeof(treenode));
    
    p->type = t;
    p->index = ind;
    p->cond1 = c;
    for (int i = 0; i < 8; i++) {
        p->treenode_children[i] = NULL;
    }
    p->exp1 = e;
    p->goodexamples = 1;
    p->badexamples = 1;
    p->next = NULL;
    p->parent = NULL;
    p->sema1 = NULL;
    if(t >= 6)
        p->fixed = 1;
    else
        p->fixed = 0;
    return p;
}

int nextrand(int range)    //generate a random number between 0..range-1
{
    int number = rand() % range;
    // printf("range=%d;number=%d;\n",range,number);
    return number;
}

exp_* genexp(program* prog,int vartype)
{
    
    int exptype = nextrand(2);
    if(vartype != -1)
        exptype = vartype;
    int varindex = 0;
    exp_* result = NULL;
    switch(exptype)
    {
        case 0:    result = createExp(exptype,nextrand(maxconst + 1));
            break;
        case 1:    varindex = nextrand(1 + numprog) - 1 - numprog;
            result = createExp(exptype,varindex);
            break;
    }
    return result;
}



exp_* genexp_(program* prog,int vartype, int index)
{
    
    int exptype = 0;
    if(vartype != -1)
        exptype = vartype;
    int varindex = 0;
    exp_* result = NULL;
    switch(exptype)
    {
        case 0:    result = createExp(exptype,index);     // 0 1
            break;
        case 1:    varindex = index;
            result = createExp(exptype,varindex);        //
            break;
    }
    return result;
}

sema* gensema()
{
    return createSema(nextrand(3) - 3);
}

sema* gensema_new(int sema)
{
    return createSema(sema - 3);
}
cond* gencond(program* prog)    //type==3:true,==,!=    type==5:true,==,!=,&&,||
{
    int condtype = nextrand(2) + 1;
    cond* result = NULL;
    switch(condtype)
    {
        case 0: result = createCond(0,NULL,NULL);break;
        case 1: result = createCond(1,genexp(prog,1),genexp(prog,-1));
            break;
        case 2: result = createCond(2,genexp(prog,1),genexp(prog,-1));
            break;
            //case 3: result = createCond(3,NULL,NULL,gencond(prog,3),gencond(prog,3));
            //        break;
            //case 4: result = createCond(4,NULL,NULL,gencond(prog,3),gencond(prog,3));
            //        break;
    }
    return result;
}




cond* gencond_(program* prog)    //type==3:true,==,!=    type==5:true,==,!=,&&,||
{
    int condtype = 1;
    cond* result = NULL;
    switch(condtype)
    {
        case 0: result = createCond(0,NULL,NULL);break;
        case 1: result = createCond(1,genexp_(prog,1,1),genexp_(prog,-1,1));
            break;
        case 2: result = createCond(2,genexp_(prog,1,1),genexp_(prog,-1,1));
            break;
            //case 3: result = createCond(3,NULL,NULL,gencond(prog,3),gencond(prog,3));
            //        break;
            //case 4: result = createCond(4,NULL,NULL,gencond(prog,3),gencond(prog,3));
            //        break;
    }
    return result;
}

//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN    5:critical section
//treenode* createTreenode(int t,int ind,cond* c,treenode* t1,treenode* t2,treenode* f,exp* e);
/*
treenode* genprog_(program* prog, int commandtype)
{
    int varindex = 0;
    
    treenode* result = NULL;
    switch(commandtype)
    {
        case 0: result = createTreenode(0,0,gencond_(prog),NULL,NULL,NULL);    //0:TRUE
            while(result->cond1->type == 0)
            {
                free(result->cond1);
                result->cond1 = gencond_(prog);
            }
            result->treenode1 = genprog_(prog, 4);
            break;
        case 1: result = createTreenode(1,0,gencond_(prog),NULL,NULL,NULL);
            while(result->cond1->type == 0)
            {
                free(result->cond1);
                result->cond1 = gencond_(prog);
            }
            result->treenode1 = genprog_(prog, 5);
            break;
            
        case 3: //varindex = nextrand(2) - 2;    //??
            varindex = -2;
            result = createTreenode(3,varindex,NULL,NULL,NULL,genexp_(prog, 0, 0));
            break;
        case 4: result = createTreenode(4,0,NULL,NULL,NULL,NULL);
            result->sema1 = createSema(-3);
            break;
        case 5:    result = createTreenode(5,0,NULL,NULL,NULL,NULL);
            result->sema1 = createSema(-2);
            break;
            
    }
    
    return result;
}
*/
/*
//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN    5:critical section
//treenode* createTreenode(int t,int ind,cond* c,treenode* t1,treenode* t2,treenode* f,exp* e);
treenode* genprog(int depth,program* prog)
{
    int commandtype;
    int height = prog->maxdepth + 1 - depth;
    
    if(height < 2)
        commandtype = nextrand(3) + 3;
    else
    {
        commandtype = nextrand(5);
        if(commandtype > 1)
            commandtype++;
    }
    
    int varindex = 0;
    
    treenode* result = NULL;
    switch(commandtype)
    {
        case 0: result = createTreenode(0,0,gencond(prog),NULL,NULL,NULL);    //0:TRUE
            while(result->cond1->type == 0)
            {
                free(result->cond1);
                result->cond1 = gencond(prog);
            }
            result->treenode1 = genprog(depth + 1,prog);
            break;
        case 1: result = createTreenode(1,0,gencond(prog),NULL,NULL,NULL);
            while(result->cond1->type == 0)
            {
                free(result->cond1);
                result->cond1 = gencond(prog);
            }
            result->treenode1 = genprog(depth + 1,prog);
            break;
        case 3: varindex = nextrand(2) - 2;    //??
            result = createTreenode(3,varindex,NULL,NULL,NULL,genexp(prog,0));
            break;
        case 4: result = createTreenode(4,0,NULL,NULL,NULL,NULL);
            result->sema1 = gensema();
            break;
        case 5:    result = createTreenode(5,0,NULL,NULL,NULL,NULL);
            result->sema1 = gensema();
            break;
            
    }
    
    return result;
}
*/

treenode* genCS(int depth,program* prog)             //??????????放到数组
{
    cond* c = createCond(-1,NULL,NULL);
    //treenode* t1 = createTreenode(2,0,NULL,NULL,NULL,NULL);
    treenode* result = createTreenode(1,0,c,NULL,NULL,NULL);
    
    result->treenode_children[0] = createTreenode(6,0,NULL,NULL,NULL,NULL);
    result->treenode_children[0]->fixed = 1;
    
    result->treenode_children[1] = createTreenode(7,0,NULL,NULL,NULL,NULL);
    result->treenode_children[1]->fixed = 1;
    
    return result;
}

//0:CONST  1:VAR  2:TIMES  3:PLUS  4:MINUS //5:DIV
void printexp(exp_* e,program* prog)
{
    switch(e->type)
    {
        case 0:    printf("%d",e->index);
            break;
        case 1:    if(e->index == -1)
            printf("v[p]");
        else if(e->index == -2)
            printf("v[me]");
        else if(e->index < -2)
            printf("v[other%d]",-e->index - 2);
        else
            printf("v[%d]",e->index);
    }
}

//0:TRUE  1:FALSE  2:EQ    3:NEQ
void printcond(cond* c,program* prog)
{
    if(c == NULL)
        return;
    switch(c->type)
    {
        case -1:printf("wi");break;
        case 0:printf("true");break;
        case 1:printexp(c->exp1,prog);printf(" == ");printexp(c->exp2,prog);break;
        case 2:printexp(c->exp1,prog);printf(" != ");printexp(c->exp2,prog);break;
    }
}
/*
//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN
treenode* findNode(treenode* root,program* prog, int num, treenode* result)
{    //printf("printprog progtype:%d blank:%d\n",prog->type,blank)
    if(root == NULL)
        return NULL;
   
    if(root->type == 2)
    {
        result = findNode(root->treenode1, prog, num, result);
        if (result == NULL)
        {
            result = findNode(root->treenode2, prog, num, result);
        }
    }
    else
    {
        if (root->number == num)
        {
            //result = copyTreenode(root);
            result = root;
        }
        else if (root->treenode1 != NULL)
        {
            result = findNode(root->treenode1, prog, num, result);
        }
    }
    
    return result;
}

*/


//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN
void printprog(treenode* root,int blank,program* prog)             // for(int i =0; i< length; i++){}
{    //printf("printprog progtype:%d blank:%d\n",prog->type,blank);
    int i;
    if(root == NULL)
        return;
    //printf("%d",root->numofstatements);
    switch(root->type)
    {
        case 0: for(i = 0;i < blank;i++)printf(" ");printf("if(");
            printcond(root->cond1,prog);
            printf(")\n");
            for(i = 0;i < blank;i++)printf(" ");printf("{\n");
            for (int index = 0; index < root->numofstatements; index++)
            {
                printprog(root->treenode_children[index],blank + 2,prog);
            }
            for(i = 0;i < blank;i++)printf(" ");printf("}\n");
            break;
        case 1: for(i = 0;i < blank;i++)printf(" ");printf("while(");
            printcond(root->cond1,prog);
            printf(")\n");
            for(i = 0;i < blank;i++)printf(" ");printf("{\n");
            // for(int i =0; i< length; i++){}
            for (int index = 0; index < root->numofstatements; index++)
            {
                printprog(root->treenode_children[index],blank + 2,prog);
            }
            for(i = 0;i < blank;i++)printf(" ");printf("}\n");
            break;
        
        case 3: for(i = 0;i < blank;i++)printf(" ");
            if(root->index >= 0)
                printf("v[%d] = ",root->index);
            else if(root->index == -1)
                printf("v[p] = ");
            else if(root->index == -2)
                printf("v[me] = ");
            else        //error
                printf("v[other%d] = ",-root->index - 2);
            printexp(root->exp1,prog);
            printf(";\n");
            break;
        case 4: for(i = 0;i < blank;i++)printf(" ");
            if(root->sema1->index == -1)
                printf("wait(mutex);\n");
            else if(root->sema1->index == -2)
                printf("wait(left);\n");
            else if(root->sema1->index == -3)
                printf("wait(right);\n");
            else
                printf("wait(sema[%d]);\n",root->sema1->index);
            break;
        case 5: for(i = 0;i < blank;i++)printf(" ");
            if(root->sema1->index == -1)
                printf("signal(mutex);\n");
            else if(root->sema1->index == -2)
                printf("signal(left);\n");
            else if(root->sema1->index == -3)
                printf("signal(right);\n");
            else
                printf("signal(sema[%d]);\n",root->sema1->index);
            break;
        case 6: for(i = 0;i < blank;i++)printf(" ");
            printf("think\n");
            break;
        case 7:for(i = 0;i < blank;i++)printf(" ");
            printf("eat\n");
            break;
    }
}

void setLinesTreenode(treenode* t,int depth)            //？？？
{
    if(t == NULL)
        return;
    
    t->depth = depth;
    
    // 遍历程序
    int height = 0;
    for (int i=0 ; i< t->numofstatements; i++)
    {
         setLinesTreenode(t->treenode_children[i], depth + 1);
    }
    
    for (int i=0 ; i< t->numofstatements; i++) {
        if (t->treenode_children[i]->height >height )
         {
            height = t->treenode_children[i]->height;
        }
    }
    t->height = height + 1;
}

/*
int setFixed(treenode* t)
{
    if(t == NULL)
        return 0;
    if(t->fixed == 1)
    {
        setFixed(t->treenode1);
        setFixed(t->treenode2);
        return 1;
    }
    
    if(t->type == 0)
    {
        int a = setFixed(t->treenode1);
        int b = setFixed(t->treenode2);
        if(a == 0 && b == 0)
            t->fixed = 0;
        else
            t->fixed = 1;
    }
    else if(t->type == 1)
        t->fixed = setFixed(t->treenode1);
    else if(t->type == 2)
    {    if(t->treenode1 == NULL || t->treenode2 == NULL)
        printf("setfixed error\n");
        int a = setFixed(t->treenode1);
        int b = setFixed(t->treenode2);
        if(a == 0 && b == 0)
            t->fixed = 0;
        else
            t->fixed = 1;
    }
    else if(t->type == 3 || t->type == 4 || t->type == 5)
        t->fixed = 0;
    else if(t->type >= 6)
        t->fixed = 1;
    
    return t->fixed;
}
*/

void setNumOfStatements(treenode* root)
{
    int numofStatements=0;
    for (int i = 0; root->treenode_children[i] != NULL; i++)
    {
        numofStatements++;
    }
    root->numofstatements = numofStatements;
    for (int i = 0; root->treenode_children[i] != NULL; i++)
    {
        setNumOfStatements(root->treenode_children[i]);
    }
}

void setAll(program* prog)
{
    prog->root->parent = NULL;
    prog->root->next = NULL;
    setNumOfStatements(prog->root);
    setNext(prog->root);//printf("x1\n");
    setParent(prog->root);//printf("x2\n");
    setLinesTreenode(prog->root,1);//printf("x5\n");
    setTreenodeNum(prog->root, prog, 0);
    
}

program** genInitTemplate(int num)
{
    program** inittemplate = (program**)malloc(sizeof(program*) * num);
    int i;
    for(i = 0;i < num;i++)
    {
        inittemplate[i] = (program*)malloc(sizeof(program));
        inittemplate[i]->maxdepth = 3;
        //inittemplate[i]->progid = j;
        inittemplate[i]->maxconst = maxconst;
        inittemplate[i]->numprivatevars = numprivatevars;
        inittemplate[i]->numpublicvars = numpublicvars;
        inittemplate[i]->root = genCS(1,inittemplate[i]);    //wrongcase1();//
        inittemplate[i]->checkedBySpin = 0;
        setAll(inittemplate[i]);
        //printf("type:%d\n\n",inittemplate[i]->root->treenode1->treenode1->treenode1->pa);
    }
    return inittemplate;
}

void setMeOther(treenode* t,cond* c,exp_* e,sema* s,int type,int progid)   //？？？
{
    //if(t!=NULL)
    //    printf("t->type:%d\n", t->type);
    //printf("type :%d\n", type);
    if(type == 1)  //treenode
    {
        if(t == NULL)
            return;
        if(t->type == 3 && t->index <= -2)
            t->index = (-t->index - 2 + progid) % numprog;
        
        //遍历prog
        for(int i = 0; i < t->numofstatements; i++)
            setMeOther(t->treenode_children[i], NULL, NULL, NULL, 1, progid);
        setMeOther(NULL,t->cond1,NULL,NULL,2,progid);
        setMeOther(NULL,NULL,t->exp1,NULL,3,progid);
        setMeOther(NULL,NULL,NULL,t->sema1,4,progid);
    }
    else if(type == 2)  //condition
    {
        if(c == NULL)
            return;
        if(c->type != 0)
        {
            setMeOther(NULL,NULL,c->exp1,NULL,3,progid);
            setMeOther(NULL,NULL,c->exp2,NULL,3,progid);
        }
    }
    else if(type == 3)     //exp
    {
        if(e == NULL)
            return;
        if(e->index <= -2)
            e->index = (-e->index - 2 + progid) % numprog;
    }
    else            //semaphore
    {
        if(s == NULL)
            return;
        if(s->index == -1)
            s->index = numprog;
        else if(s->index == -2)
            s->index = progid;
        else if(s->index == -3)
            s->index = (progid + 1) % numprog;
    }
    
}

program* genProgram(program* templat,int progid)
{
    //printf("y1\n");
    program* newprog = copyProgram(templat);
    newprog->progid = progid;
    //printf("y2\n");
    setMeOther(newprog->root,NULL,NULL,NULL,1,progid);
    //printf("y3\n");
    setAll(newprog);
    return newprog;
}

organism* genOrganism(program* templat)
{
    //printf("genOrganism\n");
    organism* result = (organism*)malloc(sizeof(organism));
    //printf("genOrganism2\n");
    result->progs = (program**)malloc(sizeof(program*) * numprog);
    //printf("genOrganism3\n");
    int i;
    //printf("x1\n");
    for(i = 0;i < numprog;i++)
    {
        //printf("genOrganism*:%d\n", i);
        result->progs[i] = genProgram(templat,i);
        //printf("genOrganism**:%d\n", i);
    }
    return result;
}

void freeAll(organism* org,program* prog,treenode* t,cond* c,exp_* e,int type)
{
    int i;
    switch(type)
    {
        case 1:    if(org == NULL)
            break;
            for(i = 0;i < numprog;i++)
                freeAll(NULL,org->progs[i],NULL,NULL,NULL,2);
            free(org);
            break;
        case 2:    if(prog == NULL)
            return;
            freeAll(NULL,NULL,prog->root,NULL,NULL,3);
            free(prog);
            break;
        case 3:    if(t == NULL)
            break;
            
            //遍历prog
            for (i = 0; i < t->numofstatements; i++)
            {
                freeAll(NULL,NULL,t->treenode_children[i],NULL,NULL,3);
            }
            
            freeAll(NULL,NULL,NULL,t->cond1,NULL,4);
            freeAll(NULL,NULL,NULL,NULL,t->exp1,5);
            free(t->sema1);
            free(t);
            break;
        case 4:    if(c == NULL)
            break;
            freeAll(NULL,NULL,NULL,NULL,c->exp1,5);
            freeAll(NULL,NULL,NULL,NULL,c->exp2,5);
            free(c);
            break;
        case 5:    if(e == NULL)
            break;
            free(e);
            break;
    }
}


void expToPml(exp_* e,FILE* f)
{
    if(e == NULL)
        return;
    
    if(e->type == 0)
    {
        if(e->index == 0)
            fputs("0",f);
        else if(e->index == 1)
            fputs("1",f);
    }
    else if(e->type == 1)
    {
        if(e->index >= 0)
            fprintf(f,"v%d",e->index);
        else if(e->index == -1)
            fputs("turn",f);
    }
}
void condToPml(cond* c,FILE* f,int blank, int progid)
{
    if(c == NULL)
        return;
    int i;
    for(i = 0;i < blank;i++)
        fputs("\t",f);
    fputs("::",f);
    
    if(c->type == -1)
    {
        fprintf(f,"wi%d == 0->\n",progid);
        //for(i = 0;i < blank + 1;i++)
        //    fputs("\t",f);
        //fprintf(f,"eat%d = 1;\n",);
        
        for(i = 0;i < blank + 1;i++)
            fputs("\t",f);
        fprintf(f,"select(wi%d:0..1);\n",progid);
    }
    else if(c->type == 0)
        fputs("true->\n",f);
    else if(c->type == 1)
    {
        expToPml(c->exp1,f);
        fputs(" == ",f);
        expToPml(c->exp2,f);
        fputs("->\n",f);
    }
    else if(c->type == 2)
    {
        expToPml(c->exp1,f);
        fputs(" != ",f);
        expToPml(c->exp2,f);
        fputs("->\n",f);
    }
}

void progToPml(treenode* t,FILE* f,program* prog,int blank) //？？
{
    if(t == NULL)
        return;
    int i;
    if(t->type == 0)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("if\n",f);
        condToPml(t->cond1,f,blank + 1,prog->progid);
        //for
        for (i = 0; t->treenode_children[i] != NULL; i++) {
            progToPml(t->treenode_children[i],f,prog,blank + 2);
        }
        
        
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("fi\n",f);
    }
    else if(t->type == 1)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("do\n",f);
        condToPml(t->cond1,f,blank + 1,prog->progid);
        //for
        for (i = 0; t->treenode_children[i] != NULL; i++) {
            progToPml(t->treenode_children[i],f,prog,blank + 2);
        }
        
        
        for(i = 0;i < blank + 1;i++)
            fputs("\t",f);
        fputs("::else->break;\n",f);
        
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("od\n",f);
    }
    else if(t->type == 3)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        if(t->index >= 0)
            fprintf(f,"v%d = ",t->index);
        else if(t->index == -1)
            fputs("turn = ",f);
        
        expToPml(t->exp1,f);
        fputs(";\n",f);
    }
    else if(t->type == 4)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("do\n",f);
        
        for(i = 0;i < blank + 1;i++)
            fputs("\t",f);
        fprintf(f,"::true->atomic{s%d==-1->s%d = %d;break;}\n",t->sema1->index,t->sema1->index,prog->progid);
        
        for(i = 0;i < blank + 1;i++)
            fputs("\t",f);
        fputs("::else->\n",f);
        
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("od\n",f);
    }
    else if(t->type == 5)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fprintf(f,"atomic{s%d==%d->s%d = -1;}\n",t->sema1->index,prog->progid,t->sema1->index);
    }
    else if(t->type == 6)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fprintf(f,"t%d = 1;\n",prog->progid);
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fprintf(f,"t%d = 0;\n",prog->progid);
        
    }
    else if(t->type == 7)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fprintf(f,"e%d = 1;\n",prog->progid);
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fprintf(f,"e%d = 0;\n",prog->progid);
        
    }
}

void orgToPml(organism* org,FILE* f)
{
    int i;
    fputs("bit turn = 0",f);
    for(i = 0;i < numprog;i++)
        fprintf(f,",v%d = 0",i);
    fputs(";\n",f);
    
    fputs("bit ",f);
    for(i = 0;i < numprog - 1;i++)
        fprintf(f,"t%d = 0,",i);
    fprintf(f,"t%d = 0;\n",i);
    
    fputs("bit ",f);
    for(i = 0;i < numprog - 1;i++)
        fprintf(f,"e%d = 0,",i);
    fprintf(f,"e%d = 0;\n",i);
    
    fputs("bit ",f);
    for(i = 0;i < numprog - 1;i++)
        fprintf(f,"wi%d = 0,",i);
    fprintf(f,"wi%d = 0;\n",i);
    
    fputs("int ",f);
    for(i = 0;i < numprog;i++)
        fprintf(f,"s%d = -1,",i);
    fprintf(f,"s%d = -1;\n",i);
    
    
    for(i = 0;i < numprog;i++)
    {
        fprintf(f,"active proctype p%d()\n{\n\tselect(wi0:0..1);\n",i);
        progToPml(org->progs[i]->root,f,org->progs[i],1);
        fputs("}\n\n",f);
    }
    
    
    for(i = 0;i < numprog;i++)
    {
        fprintf(f,"ltl e%d{[]((t%d == 1) -> <>(e%d == 1))}\n",i * 2 + 1,i,i);
        fprintf(f,"ltl e%d{[]((e%d == 1) -> ((s%d == %d) && (s%d == %d)))}\n",i * 2 + 2,i,i,i,(i + 1) % numprog,i);
    }
}

void expTorml(exp_* e,FILE *f)
{
    if(e == NULL)
        return;
    
    if(e->type == 0)
        fprintf(f,"%d",e->index);
    else if(e->type == 1)
    {
        if(e->index >= 0)
            fprintf(f,"v%d",e->index);
        else if(e->index == -1)
            fputs("turn",f);
    }
}


void condTorml(cond* c,FILE *f)
{
    if(c == NULL)
        return;
    
    if(c->type == 0 || c->type == -1)
        fputs("true",f);
    else if(c->type == 1)
    {
        expTorml(c->exp1,f);
        fputs("=",f);
        expTorml(c->exp2,f);
    }
    else if(c->type == 2)
    {
        expTorml(c->exp1,f);
        fputs("!=",f);
        expTorml(c->exp2,f);
    }
}

//mutation.c

int mutype;
sema* copySema(sema* s)
{
    if(s == NULL)
        return NULL;
    sema* result = (sema*)malloc(sizeof(sema));
    result->index = s->index;
    return result;
}

exp_* copyExp(exp_* e)
{
    if(e == NULL)
        return NULL;
    exp_* result = (exp_*)malloc(sizeof(exp_));
    result->type = e->type;
    result->index = e->index;
    return result;
}

cond* copyCond(cond* c)
{
    if(c == NULL)
        return NULL;
    cond* result = (cond*)malloc(sizeof(cond));
    result->type = c->type;
    result->exp1 = copyExp(c->exp1);
    result->exp2 = copyExp(c->exp2);
    return result;
}

treenode* copyTreenode(treenode* t)
{
    if(t == NULL)
        return NULL;
    treenode* result = (treenode*)malloc(sizeof(treenode));
    result->type = t->type;
    result->index = t->index;
    result->cond1 = copyCond(t->cond1);
    
    for (int i = 0; i < 8; i++)
    {
        result->treenode_children[i] = copyTreenode(t->treenode_children[i]);
    }
    result->exp1 = copyExp(t->exp1);
    result->sema1 = copySema(t->sema1);
    result->goodexamples = t->goodexamples;
    result->badexamples = t->badexamples;
    result->depth = t->depth;
    result->height = t->height;
    result->fixed = t->fixed;
    result->number = t->number;
    result->numofstatements = t->numofstatements;        //!!
    //setParent(result);
    return result;
}

program* copyProgram(program* prog)
{
    program* result = (program*)malloc(sizeof(program));
    result->root = copyTreenode(prog->root);
    setParent(result->root);
    result->maxdepth = prog->maxdepth;
    result->progid = prog->progid;
    result->maxconst = prog->maxconst;
    result->numprivatevars = prog->numprivatevars;
    result->numpublicvars = prog->numpublicvars;
    result->checkedBySpin = prog->checkedBySpin;
    result->fitness = prog->fitness;
    return result;
}

void printUnOp(UnOp op)
{
    switch(op)
    {
        case Neg:printf("~");break;
        case Not:printf("!");break;
        case Future:printf("F");break;
        case Globally:printf("G");break;
        case Next:printf("X");break;
    }
}

void printBinOp(BinOp op)
{
    switch(op)
    {
        case Imp:printf("->");break;
        case And:printf("&");break;
        case Or:printf("|");break;
        case Eq:printf("=");break;
        case Neq:printf("!=");break;
        case Lt:printf("<");break;
        case Le:printf("<=");break;
        case Gt:printf(">");break;
        case Ge:printf(">=");break;
        case Add:printf("+");break;
        case Min:printf("-");break;
        case Mul:printf("*");break;
        case Div:printf("/");break;
    }
}

void printExpr(Expr* exp)
{
    if(exp == NULL)
        return;
    
    switch(exp->type)
    {
        case Constant:printf("%d",exp->value);break;
        case Variable:printf("%s",exp->name);break;
        case Expr1:printUnOp(exp->uop);printExpr(exp->child);break;
        case StepBoundExpr1:printf("(");printUnOp(exp->uop);printf("<=#%d",exp->stepbound);printExpr(exp->child);printf(")");break;
        case Expr2:printf("(");printExpr(exp->left);printBinOp(exp->bop);printExpr(exp->right);printf(")");break;
    }
}

/*
int satisfyMutationReduction(treenode* t)        //!!!
{
    if(t->treenode1 == NULL && t->treenode2 == NULL)
        return 0;
    if(t->type == 0)
    {
        if(getFixed(t->treenode1) == 1 || getFixed(t->treenode2) == 1)
            printf("if child fixed error!!\n");
        return 1;
    }
    else if(t->type == 1)
    {
        if(t->fixed == 0 && t->treenode1 != NULL)
            return 1;
    }
    else if(t->type == 2)
    {
        //if(getFixed(t->treenode1) == 0 || getFixed(t->treenode2) == 0)    //||
        return 1;
    }
    return 0;
}
*/

void mutationCond(cond* root,program* prog,int type)    //type == 1:can add        type == 0:can't add
{
    if(root == NULL)
        return;
    if(root->type == -1)
        return;
    if(root->type == 0)
    {
        printf("ERROR:CONDITION TYPE 0!!!\n");
        cond* new_ = gencond(prog);
        while(new_->type == 0)
        {
            free(new_);
            new_ = gencond(prog);
        }
        root->type = new_->type;
        root->exp1 = new_->exp1;
        root->exp2 = new_->exp2;
        free(new_);
    }
    else if(root->type == 1 || root->type == 2)
    {
        int t = nextrand(4);//printf("a%d",t);
        if(t == 0)        //change left
            root->exp1 = genexp(prog,1);
        else if(t == 1)        //change right
            root->exp2 = genexp(prog,0);
        else if(t == 2)        //change == / !=
            root->type = 3 - root->type;
        else if(t == 3)                //change entirely
        {
            free(root->exp1);
            free(root->exp2);
            
            cond* new_ = gencond(prog);
            root->type = new_->type;
            root->exp1 = new_->exp1;
            root->exp2 = new_->exp2;
        }
    }
}

/*
treenode* getStatement(treenode* seq, int t)    //t=0:first    t=1:last
{
    if(seq == NULL || seq->type != 3 || seq->depth != 2)
        return seq;
    
    if(t == 0)
        return getStatement(seq->treenode1,0);
    else
        return getStatement(seq->treenode2,1);
}
*/

/*
int* legalAction2(program* parent, int nodeNum)
{
    int* action = (int*)malloc(sizeof(int)*27);
    memset(action, 0, sizeof(int)*27);
    program* newprog = copyProgram(parent);
    treenode* new = newprog->root;
    treenode* chnode = NULL;
    chnode = findNode(newprog->root, newprog, nodeNum, chnode);
    if (chnode == NULL) {
        newprog->illegal = 1;
    }
    treenode* mnode = chnode;
    int i_act = 0;
    if((mnode->type == 0 || mnode->type == 1))
       action[i_act] = 0;
    else
        for (int k = 1; k < 6; k++)
        {
            action[i_act] = k;
            i_act++;
        }
    
    if (mnode->type <= 1 && (mnode->cond1->type == 1 || mnode->cond1->type == 2))
    {
        for (int k = 6; k < 14; k++)
        {
            action[i_act] = k;
            i_act++;
        }
    }
    else if(mnode->type == 3)
    {
        for (int k = 14; k < 17; k++)
        {
            action[i_act] = k;
            i_act++;
        }
    }
    else if(mnode->type == 4 || mnode->type == 5)
    {
        for (int k = 17; k < 20; k++)
        {
            action[i_act] = k;
            i_act++;
        }
    }
    
    if (mnode->depth + mnode->height == newprog->maxdepth + 1)
    {
        for (int k = 22; k < 24; k++)
        {
            action[i_act] = k;
            i_act++;
        }
    }
    else if (mnode->depth == 2 && mnode->numofstatements >= 8 || mnode->depth == 3 && mnode->numofstatements >= 4)
    {
        for (int k = 20; k < 22; k++)
        {
            action[i_act] = k;
            i_act++;
        }
    }
    else
    {
        for (int k = 20; k < 24; k++)
        {
            action[i_act] = k;
            i_act++;
        }
    }
    
    
    if (mnode->treenode1 != NULL && mnode->treenode1->fixed != 1 && mnode->fixed != 1 && (mnode->treenode2 == NULL || mnode->treenode2->fixed == 0))
    {
        action[i_act] = 24;
        i_act++;
    }
    else if(mnode->treenode2 != NULL && mnode->treenode2->fixed != 1 && mnode->fixed != 1 &&  mnode->treenode1->fixed == 0)
    {
        action[i_act] = 25;
        i_act++;
    }
    
    if(mnode->parent != NULL && mnode->parent->type == 2 && mnode->fixed != 1)
    {
        action[i_act] = 26;
        i_act++;
    }
    
    
    printf("\n");
    for (int i =0; i < 26; i++) {
        printf("%d ", action[i]);
    }
    return action;
}
*/

/*
program *mutation_(program* parent, int nodeNum, int actionNum)
{
    program* newprog = copyProgram(parent);
    newprog->checkedBySpin = 0;
    treenode* new_ = newprog->root;
    treenode* chnod = NULL;
    int temp;
    chnod = findNode(newprog->root, newprog, nodeNum, chnod);    
    treenode* mnode = chnod;
    //printprog(chnod, 0, newprog);
    treenode* newnode;
    exp_* e;
    switch (actionNum) {
        // replacement treenode
        case 0:
            mnode->type = 1 - mnode->type;
            break;
        case 1:
            newnode = genprog_(newprog, 0);          // if
            
            if(mnode->parent == NULL)
                new_ = newnode;
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                free(mnode);
            }
            break;
        case 2:
            newnode = genprog_(newprog, 1);           // while
            if(mnode->parent == NULL)
                new_ = newnode;
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                free(mnode);
            }
            break;
        case 3:
            newnode = genprog_(newprog, 3);           //assign
            if(mnode->parent == NULL)
                new_ = newnode;
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                free(mnode);
            }
            break;
        case 4:
            newnode = genprog_(newprog, 4);          //wait
            if(mnode->parent == NULL)
                new_ = newnode;
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                free(mnode);
            }
            break;
        case 5:
            newnode = genprog_(newprog, 5);          //signal
            if(mnode->parent == NULL)
                new_ = newnode;
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                free(mnode);
            }
            break;
        // replace cond if/while
        case 6:
            mnode->cond1->exp1 = genexp_(newprog, 1, -4);
            break;
        case 7:
            mnode->cond1->exp1 = genexp_(newprog, 1, -3);
            break;
        case 8:
            mnode->cond1->exp1 = genexp_(newprog, 1, -2);
            break;
        case 9:
            mnode->cond1->exp1 = genexp_(newprog, 1, -1);
            break;
        case 10:
            mnode->cond1->exp2 = genexp_(newprog, 0, 0);
            break;
        case 11:
            mnode->cond1->exp2 = genexp_(newprog, 0, 1);
            break;
        case 12:
            free(mnode->cond1->exp1);
            free(mnode->cond1->exp2);
            cond* new_1 = gencond_(newprog);
            mnode->cond1->type = new_1->type;
            mnode->cond1->exp1 = new_1->exp1;
            mnode->cond1->exp2 = new_1->exp2;
            break;
        case 13:
            mnode->cond1->type = 3 - mnode->cond1->type;
            break;
        // replace assign
        case 14:
            e = genexp_(newprog,0,0);
            free(mnode->exp1);
            mnode->exp1 = e;
            break;
        case 15:
            e = genexp_(newprog,0,1);
            free(mnode->exp1);
            mnode->exp1 = e;
            break;
        case 16:
            mnode->index = -3 - mnode->index;
            break;
        // wait signal
        case 17:
            temp = -3;
            mnode->sema1->index = temp;
            break;
        case 18:
            temp = -2;
            mnode->sema1->index = temp;
            break;
        case 19:
            temp = -1;
            mnode->sema1->index = temp;
            break;
        //insert
        case 20:
            newnode = createTreenode(0,0,NULL,NULL,NULL,NULL);  //if
            newnode->cond1 = gencond_(newprog);
            newnode->treenode1 = mnode;
            if(mnode->parent == NULL)
            {
                mnode->parent = newnode;
                new_ = newnode;
            }
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else if(mnode->parent->treenode2 == mnode)
                    mnode->parent->treenode2 = newnode;
                else
                    printf("mutation type2 error!\n");
                
                newnode->parent = mnode->parent;
                mnode->parent = newnode;
            }
            break;
        case 21:
            newnode = createTreenode(1,0,NULL,NULL,NULL,NULL);  //while
            newnode->cond1 = gencond_(newprog);
            newnode->treenode1 = mnode;
            if(mnode->parent == NULL)
            {
                mnode->parent = newnode;
                new_ = newnode;
            }
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else if(mnode->parent->treenode2 == mnode)
                    mnode->parent->treenode2 = newnode;
                else
                    printf("mutation type2 error!\n");
                
                newnode->parent = mnode->parent;
                mnode->parent = newnode;
            }
            break;
        case 22:
            newnode = createTreenode(2,0,NULL,NULL,NULL,NULL);
            newnode->treenode1 = mnode;
            newnode->treenode2 = genprog_(newprog, 4);
            if(mnode->parent == NULL)
            {
                mnode->parent = newnode;
                new_ = newnode;
            }
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                mnode->parent = newnode;
            }
            break;
        case 23:
            newnode = createTreenode(2,0,NULL,NULL,NULL,NULL);
            newnode->treenode2 = mnode;
            newnode->treenode1 = genprog_(newprog, 5);
            if(mnode->parent == NULL)
            {
                mnode->parent = newnode;
                new_ = newnode;
            }
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                mnode->parent = newnode;
            }
            break;
        case 24:
            // reduction
            printf("\n%d", actionNum);
            if(mnode->parent == NULL)    //mnode == new
            {
                new_ = mnode->treenode1;
                new_->parent = NULL;
            }
            else                         //mnode != new
            {
                mnode->treenode1->parent = mnode->parent;
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = mnode->treenode1;
                else
                    mnode->parent->treenode2 = mnode->treenode1;
            }
            free(mnode);
            break;
        case 25:
            if(mnode->parent == NULL)    //mnode == new
            {
                new_ = mnode->treenode2;
                new_->parent = NULL;
            }
            else                         //mnode != new
            {
                mnode->treenode2->parent = mnode->parent;
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = mnode->treenode2;
                else
                    mnode->parent->treenode2 = mnode->treenode2;
            }
            free(mnode);
            break;
        case 26:
            if (mnode->parent->treenode1 == mnode)
            {
                if(mnode->parent == mnode->parent->parent->treenode1)
                {
                    mnode->parent->parent->treenode1 = mnode->parent->treenode2;
                    mnode->parent->treenode2->parent = mnode->parent->parent;
                }
                else if(mnode->parent == mnode->parent->parent->treenode2)
                {
                    mnode->parent->parent->treenode2 = mnode->parent->treenode2;
                    mnode->parent->treenode2->parent = mnode->parent->parent;
                }
                freeAll(0, 0, mnode, 0, 0, 0);
            }
            else if(mnode->parent->treenode2 == mnode)
            {
                if(mnode->parent == mnode->parent->parent->treenode1)
                {
                    mnode->parent->parent->treenode1 = mnode->parent->treenode1;
                    mnode->parent->treenode1->parent = mnode->parent->parent;
                }
                else if(mnode->parent == mnode->parent->parent->treenode2)
                {
                    mnode->parent->parent->treenode2 = mnode->parent->treenode1;
                    mnode->parent->treenode2->parent = mnode->parent->parent;
                }
                freeAll(0, 0, mnode, 0, 0, 0);
            }
            break;
    }
    newprog->root = new_;
    //printf("endofmutation\n");
    return newprog;
}
*/

TREENODENUMBER = 5;
CONDNUMBER = 2;
VARIABLENUMBER = 7;
CONSTNUMBER = 9;


/*
program* mutation(program* parent)
{    //printf("startofmutation");
    program* newprog = copyProgram(parent);
    newprog->checkedBySpin = 0;
    treenode* new = newprog->root;
    
    //printf("mutationtype = %d\n",mutationtype);
    int mutationtype;
    mutationNode* chnode;
    
    do
    {
        treeRank* tr = (treeRank*)malloc(sizeof(treeRank));
        tr->candidate = (mutationNode**)malloc(sizeof(mutationNode*) * 10);
        tr->numcandidate = 0;
        tr->maxnumcandidate = 10;
        tr->ranksum = 0;
        mutationtype = nextrand(4) + 1;
        searchNode(new,tr,mutationtype,newprog->maxdepth);
        chnode = chooseNode(tr);
        //printf("mutationtype:%d,treeranksum:%d\n",mutationtype,tr->ranksum);
    }while(chnode == NULL);
    
    //printf("mutype=%d,choosenodetype:%d,nodedepth:%d\n",mutationtype,chnode->node->type,chnode->node->depth);
    mutype = mutationtype;
    if(mutationtype == 1)                            //Replacement Mutation type
    {
        treenode* mnode = chnode->node;
        free(chnode);
        if(mnode->cond1 == NULL && mnode->exp1 == NULL && mnode->sema1 == NULL || nextrand(2) == 0)
        {
            if((mnode->type == 0 || mnode->type == 1) && nextrand(2) == 0)
                mnode->type = 1 - mnode->type;
            else
            {
                treenode* newnode;
                //if(mnode->depth != 2)                    //limit heights of statements(depth=2) can only be 1.(assignment,wait,signal)
                newnode = genprog(mnode->depth,newprog);
                //else
                //    newnode = genprog(newprog->maxdepth,newprog);
                if(mnode->parent == NULL)
                    new = newnode;
                else
                {
                    if(mnode->parent->treenode1 == mnode)
                        mnode->parent->treenode1 = newnode;
                    else
                        mnode->parent->treenode2 = newnode;
                    newnode->parent = mnode->parent;
                    free(mnode);
                }
            }
        }
        else
        {
            if(mnode->type <= 1)
                mutationCond(mnode->cond1,newprog,1);
            else if(mnode->type == 3)
            {
                if(nextrand(2) == 0)
                {
                    exp_* e;
                    do
                    {
                        e = genexp(newprog,0);
                    }while(equalExp(e,mnode->exp1) == 1);
                    free(mnode->exp1);
                    mnode->exp1 = e;
                    
                }
                else
                {
                    mnode->index = -3 - mnode->index;    //-3 = v[p] + v[me]
                }
            }
            else if(mnode->type == 4 || mnode->type == 5)
            {
                int temp;
                do
                {
                    temp = nextrand(3) - 3;
                }while(mnode->sema1->index == temp);
                mnode->sema1->index = temp;
            }
        }
    }
    else if(mutationtype == 2)                        //Insert Mutation types
    {
        treenode* mnode = chnode->node;
        free(chnode);
        
        int t;
        //if(t == 2)
        //    t = 3;
        if(mnode->depth + mnode->height == newprog->maxdepth + 1)
            t = 2;
        else if(mnode->depth == 2 && mnode->numofstatements >= 8 || mnode->depth == 3 && mnode->numofstatements >= 4)
            t = nextrand(2);
        else
            t = nextrand(3);
        //if(t == 2 && (mnode->depth == 2 && mnode->numofstatements >= 8 || mnode->depth == 3 && mnode->numofstatements >= 2))
        //    printf("ERROR:add seq.depth:%d,numofstatements:%d\n",mnode->depth,mnode->numofstatements);
        treenode* newnode = createTreenode(t,0,NULL,NULL,NULL,NULL);
        
        if(t == 0 || t == 1)    //if,while
        {
            newnode->cond1 = gencond(newprog);
            newnode->treenode1 = mnode;
            if(mnode->parent == NULL)
            {
                mnode->parent = newnode;
                new = newnode;
            }
            else
            {
                
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else if(mnode->parent->treenode2 == mnode)
                    mnode->parent->treenode2 = newnode;
                else
                    printf("mutation type2 error!\n");
                
                newnode->parent = mnode->parent;
                mnode->parent = newnode;
            }
        }
        else
        {
            int p;
 
            p = nextrand(2);
            
            if(p == 0)
            {
                newnode->treenode1 = mnode;
                if(mnode->depth == 2)
                    newnode->treenode2 = genprog(newprog->maxdepth,newprog);
                else
                    newnode->treenode2 = genprog(mnode->depth,newprog);
            }
            else
            {
                newnode->treenode2 = mnode;
                if(mnode->depth == 2)
                    newnode->treenode1 = genprog(newprog->maxdepth,newprog);
                else
                    newnode->treenode1 = genprog(mnode->depth,newprog);
            }
            
            if(mnode->parent == NULL)
            {
                mnode->parent = newnode;
                new = newnode;
            }
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                mnode->parent = newnode;
            }
        }
        //printf("newnodetype%d\n",newnode->type);
    }
    else if(mutationtype == 3)                        //Reduction Mutation type
    {
        treenode* mnode = chnode->node;
        free(chnode);
        int child = nextrand(2);
        if(mnode->treenode1->fixed == 1 || mnode->treenode2 == NULL || child == 0 && mnode->treenode2->fixed == 0)//treenode1 replace mnode
        {
            if(mnode->parent == NULL)    //mnode == new
            {
                new = mnode->treenode1;
                new->parent = NULL;
            }
            else                         //mnode != new
            {
                mnode->treenode1->parent = mnode->parent;
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = mnode->treenode1;
                else
                    mnode->parent->treenode2 = mnode->treenode1;
            }
            free(mnode);
        }
        else
        {
            if(mnode->parent == NULL)    //mnode == new
            {
                new = mnode->treenode2;
                new->parent = NULL;
            }
            else                         //mnode != new
            {
                mnode->treenode2->parent = mnode->parent;
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = mnode->treenode2;
                else
                    mnode->parent->treenode2 = mnode->treenode2;
            }
            free(mnode);
        }
    }
    else                            //Deletion mutation type
    {
        treenode* mnode = chnode->node;
        free(chnode);
        //printf("mnode depth:%d,mnode type%d\n",mnode->depth,mnode->type);
        if(mnode->treenode1 == NULL)
            mnode->treenode1 = genprog(mnode->depth + 1,newprog);
        else
        {
            free(mnode->treenode1);
            mnode->treenode1 = NULL;
        }
    }
    newprog->root = new;
    //printf("endofmutation\n");
    return newprog;
}
*/

/*
bool statementsTooLong(treenode* t)
{
    if(t == NULL)
        return false;
    if(t->numofstatements > 3)
        return true;
    if(statementsTooLong(t->treenode1))
        return true;
    if(statementsTooLong(t->treenode1))
        return true;
    return false;
}
*/
/*
int numOfWi(treenode* t)
{
    if(t == NULL)
        return 0;
    int a = 0,b = 0,c = 0;
    if(t->type == 1 && t->cond1->type == -1)
        a = 1;
    b = numOfWi(t->treenode1);
    c = numOfWi(t->treenode2);
    
    return (a + b + c);
}
*/

/*bool existNullCond(treenode* t)
 {
 if(t == NULL)
 return false;
 if(t->type == 0 || t->type == 1)
 if(t->cond1 == NULL || (t->cond1->type == 3 || t->cond1->type == 4) && (t->cond1->cond1 == NULL || t->cond1->cond2 == NULL))
 return true;
 return (existNullCond(t->treenode1) || existNullCond(t->treenode2));
 }
 
 bool existMultiCond(treenode* t)
 {
 if(t == NULL)
 return false;
 if(t->type == 0 || t->type == 1)
 {
 if(t->cond1->cond1 != NULL && (t->cond1->cond1->type == 3 || t->cond1->cond1->type == 4) )
 return true;
 if(t->cond1->cond2 != NULL && (t->cond1->cond2->type == 3 || t->cond1->cond2->type == 4))
 return true;
 }
 return (existMultiCond(t->treenode1) || existMultiCond(t->treenode2));
 } */

/*
bool existdepth2if(treenode* t)
{
    if(t == NULL)
        return false;
    
    if(t->depth == 1)
        return (existdepth2if(t->treenode1) || existdepth2if(t->treenode2));
    
    if(t->depth == 2)
        if(t->type <= 1 && t->fixed == 0)
            return true;
    return false;
}
*/

program** genNewCandidate(int numofcandidate,program** candidate,int numofmutation)
{
    int* selected = (int*)malloc(sizeof(int) * numofcandidate);
    program** result = (program**)malloc(sizeof(program*) * numofmutation);
    int i;
    for(i = 0;i < numofcandidate;i++)
        selected[i] = 0;
    
    int count = 0;
    while(count < numofmutation)
    {
        int index;
        do
        {
            index = nextrand(numofcandidate);
        }while(selected[index] == 1);
        
        selected[index] = 1;
        count++;
    }
    //printf("Choose candidate:");
    //for(i = 0;i < numofcandidate;i++)
    //    printf("%d:%d;     ",i,selected[i]);
    //printf("\n");
    count = 0;
    for(i = 0;i < numofcandidate;i++)
    {
        if(selected[i] == 1)
        {
            //!!result[count] = (program*)malloc(sizeof(program));
            
            
            if(candidate[i] == NULL)printf("gennewcandidate error candidate null\n");
            
            //printType(candidate[i]->progs[1]->root);
            //printprog(candidate[i]->progs[1]->root,0);
            //printf("before mutation\n");
            //result[count] = mutation(candidate[i]);//printf("after mutation\n");
            //if(!statementsTooLong(candidate[i]->root) && statementsTooLong(result[count]->root))
            //if(numOfWi(candidate[i]->root) == 1 && numOfWi(result[count]->root) != 1)
            /*
            if(!existdepth2if(candidate[i]->root) && existdepth2if(result[count]->root))
            {
                printf("mutype = %d\nbefore mutation:\n",mutype);
                printprog(candidate[i]->root,0,candidate[i]);
                printf("after mutation\n");
                printprog(result[count]->root,0,result[count]);
                exit(-1);
            }
             */
            //printf("after mutation\n");
            /*if(!existdepth2if(candidate[i]->progs[1]->root) && existdepth2if(result[count]->progs[1]->root))
             {
             printf("CS disappear\nbefore prog:\n");
             printType(candidate[i]->->root);printf("\n");
             setFixed(candidate[i]->progs[1]->root);
             printType(candidate[i]->progs[1]->root);printf("\n");
             printprog(candidate[i]->progs[1]->root,0,candidate[i]->progs[1]);
             printf("after prog:\n");
             printprog(result[count]->progs[1]->root,0,result[count]->progs[1]);
             }*/
            
            //printf("xxx\n");
            setAll(result[count]);
            //printf("yyy\n");
            count++;
        }
    }
    free(selected);
    
    return result;
}


program** selectNewCandidate(int numofcandidate,program** candidate,int numofmutation, program** newcandidate,double* bord)
{
    program** output = (program**)malloc(sizeof(program*) * numofcandidate);
    double* f = (double*)malloc(sizeof(double) * (numofcandidate + numofmutation));
    int *chosen1 = (int*)malloc(sizeof(int) * numofcandidate);
    int *chosen2 = (int*)malloc(sizeof(int) * numofmutation);
    int i,j;
    for(i = 0;i < numofcandidate;i++)
    {
        f[i] = candidate[i]->fitness;
        chosen1[i] = 0;
    }
    for(i = 0;i < numofmutation;i++)
    {
        f[i + numofcandidate] = newcandidate[i]->fitness;
        chosen2[i] = 0;
    }
    for(i = 0;i < numofcandidate + numofmutation - 1;i++)
        for(j = 0;j < numofcandidate + numofmutation - 1 - i;j++)
            if(f[j] < f[j + 1])
            {
                f[j] = f[j] + f[j + 1];
                f[j + 1] = f[j] - f[j + 1];
                f[j] = f[j] - f[j + 1];
            }
    double border = f[numofcandidate - 1];
    *bord = border;
    //for(i = 0;i < numofcandidate;i++) printf("f[%d]->f=%lf  ",i,candidate[i]->fitness);
    //for(i = 0;i < numofmutation;i++) printf("f[%d]->f=%lf  ",i,newcandidate[i]->fitness);
    printf("border=%lf\n",border);
    int count = 0;
    //printf("Choose new candidate:\n");
    for(i = 0;i < numofcandidate;i++)
        if(candidate[i]->fitness > border + 0.0001)
        {    //printf("candidate%d,",i);
            output[count] = candidate[i];
            count++;
            chosen1[i] = 1;
        }
    for(i = 0;i < numofmutation;i++)
        if(newcandidate[i]->fitness > border + 0.0001)
        {    //printf("new candidate%d,",i);
            output[count] = newcandidate[i];
            count++;
            chosen2[i] = 1;
        }
    
    for(i = 0;i < numofcandidate && count < numofcandidate;i++)
        if(candidate[i]->fitness > border - 0.0001 && candidate[i]->fitness < border + 0.0001)
        {    //printf("candidate%d,",i);
            output[count] = candidate[i];
            count++;
            chosen1[i] = 1;
        }
    for(i = 0;i < numofmutation && count < numofcandidate;i++)
        if(newcandidate[i]->fitness > border - 0.0001 && newcandidate[i]->fitness < border + 0.0001)
        {    //printf("new candidate%d,",i);
            output[count] = newcandidate[i];
            count++;
            chosen2[i] = 1;
        }
    //printf("ccount=%d",count);
    
    for(i = 0;i < numofcandidate;i++)
        if(chosen1[i] == 0)
            freeAll(NULL,candidate[i],NULL,NULL,NULL,2);
    for(i = 0;i < numofmutation;i++)
        if(chosen2[i] == 0)
            freeAll(NULL,newcandidate[i],NULL,NULL,NULL,2);
    free(f);
    free(chosen1);
    free(chosen2);
    return output;
}

/*
program** genNewCandidateWithCoefficient1(int numofcandidate,program** candidate,int numofmutation,double coef)//no repeat
{
    int* selected = (int*)malloc(sizeof(int) * numofcandidate);
    program** result = (program**)malloc(sizeof(program*) * numofmutation);
    int i;
    for(i = 0;i < numofcandidate;i++)
        selected[i] = 0;
    
    int count = 0;
    while(count < numofmutation)
    {
        int index;
        do
        {
            index = nextrand(numofcandidate);
        }while(selected[index] == 1);
        
        selected[index] = 1;
        count++;
    }
    
    count = 0;
    for(i = 0;i < numofcandidate;i++)
    {
        if(selected[i] == 1)
        {
            
            if(candidate[i] == NULL)printf("gennewcandidate error candidate null\n");
            
            if(nextrand(100) < coef * 100)
                result[count] = mutation(candidate[i]);
            else
                result[count] = copyProgram(candidate[i]);
            
            setAll(result[count]);
            count++;
        }
    }
    free(selected);
    
    return result;
}

program** genNewCandidateWithCoefficient2(int numofcandidate,program** candidate,int numofmutation,double coef)//with repeat
{
    
    program** result = (program**)malloc(sizeof(program*) * numofmutation);
    int count = 0;
    
    while(count < numofmutation)
    {
        int index = nextrand(numofcandidate);
        
        if(nextrand(100) < coef * 100)
            result[count] = mutation(candidate[index]);
        else
            result[count] = copyProgram(candidate[index]);
        
        setAll(result[count]);
        count++;
    }
    
    return result;
}
*/
program** selectNewCandidateWithFitness(int numofcandidate,program** candidate,int numofmutation, program** newcandidate)
{
    
    program** output = (program**)malloc(sizeof(program*) * numofcandidate);
    int* f = (int*)malloc(sizeof(double) * (numofcandidate + numofmutation));
    int *chosen = (int*)malloc(sizeof(int) * (numofcandidate + numofmutation));
    
    int addfitness = 0;
    int i,j;
    for(i = 0;i < numofcandidate;i++)
    {
        f[i] = candidate[i]->fitness;
        chosen[i] = 0;
        addfitness += f[i];
    }
    for(i = 0;i < numofmutation;i++)
    {
        f[i + numofcandidate] = newcandidate[i]->fitness;
        chosen[i + numofcandidate] = 0;
        addfitness += f[i + numofcandidate];
    }
    
    int* roulette = (int*)malloc(sizeof(int) * addfitness);
    int tempfit = 0;
    for(i = 0;i < numofcandidate;i++)
        for(j = 0;j < f[i];j++)
        {
            //printf("%dtempfit = %d,i=%d,j=%d\n",addfitness,tempfit,i,j);
            roulette[tempfit++] = i;
        }
    for(i = 0;i < numofmutation;i++)
    {
        for(j = 0;j < f[i + numofcandidate];j++)
        {
            //printf("%dtemfit = %d,i=%d,j=%d\n",addfitness,tempfit,i,j);
            roulette[tempfit++] = i + numofcandidate;
        }
    }
    
    if(tempfit != addfitness)
        printf("ERROR:TEMPFIT%d != ADDFITESS%d\n",tempfit,addfitness);
    
    int count = 0;
    while(count < numofcandidate)
    {
        int p = nextrand(addfitness);
        if(chosen[roulette[p]] == 0)
        {
            chosen[roulette[p]] = 1;
            count++;
        }
    }
    
    count = 0;
    for(i = 0;i < numofcandidate;i++)
    {
        if(chosen[i] == 1)
        {
            output[count] = candidate[i];
            count++;
        }
        else
            freeAll(NULL,candidate[i],NULL,NULL,NULL,2);
    }
    for(i = 0;i < numofmutation;i++)
    {
        if(chosen[i + numofcandidate] == 1)
        {
            output[count] = newcandidate[i];
            count++;
        }
        else
            freeAll(NULL,newcandidate[i],NULL,NULL,NULL,2);
    }
    
    free(f);
    free(chosen);
    free(roulette);
    
    return output;
}

//trace.c

int *varvalue;
int *semavalue;
treenode** nextstep;
treenode** currentstep;
trace* gtrace = NULL;

int numofcheck = 300;
int numpublicvars = 1;
int numprivatevars = 1;
int maxconst = 1;

int mutype;
int condnull = 0;
void freeTrace(trace* t)
{
    int i;
    for(i = 0;i < t->steplength;i++)
        free(t->valueofsema[i]);
    free(t->valueofsema);
    free(t->executeprogid);
    free(t->executenode);
    free(t);
}

void setbadexamples(trace** traces,int num)
{
    
    int i,j;
    for(i = 0;i < num;i++)
    {
        for(j = 0;j < traces[i]->steplength;j++)
        {
            if(traces[i]->satisfied)
                traces[i]->executenode[j]->goodexamples++;
            else
                traces[i]->executenode[j]->badexamples++;
        }
        
    }
}

void setNext(treenode* root)         //???
{
    if(root == NULL)
        return;
    int i;
    if(root->type == 0 && root->treenode_children[0] != NULL)
    {
        
        i = 0;
        while(root->treenode_children[i] != NULL && root->treenode_children[i+1] != NULL)
        {
            
            root->treenode_children[i]->next = root->treenode_children[i+1];
            i++;
        }
        root->treenode_children[i]->next = root->next;
        
    }
    else if(root->type == 1 && root->treenode_children[0] != NULL)
    {
        i = 0;
        while(root->treenode_children[i] != NULL && root->treenode_children[i+1] != NULL)
        {
            
            root->treenode_children[i]->next = root->treenode_children[i+1];
            i++;
        }
        root->treenode_children[i]->next = root;
    }
    
    //for
    for (i = 0; i < root->numofstatements; i++)
    {
        setNext(root->treenode_children[i]);
    }
    
}

void setParent(treenode* root)
{
    if(root != NULL)
    {
        // for
        for (int i = 0; i < root->numofstatements; i++)
        {
            root->treenode_children[i]->parent = root;
            setParent(root->treenode_children[i]);
        }
    }
}

int getsemavalue(sema* s)
{
    if(s == NULL)
    {
        printf("getsemavalue null!\n");
        return -2;
    }
    //printf("getsemavalue.index=%d,value=%d\n",s->index,semavalue[s->index]);
    return semavalue[s->index];
}

int getvarvalue(exp_* e)
{
    if(e == NULL)
    {
        printf("getvarvalue null!\n");
        return -1;
    }
    
    if(e->type == 0)
        return e->index;
    if(e->type == 1)
        return varvalue[e->index + 1];
    return 0;
}

bool getcondvalue(cond* c)
{
    if(c == NULL)
    {
        printf("getcondvalue null!\n");
        condnull = 1;
        return false;
    }
    if(c->type == -1)
        printf("condition type = -1\n");
    switch(c->type)
    {
        case -1:if(nextrand(10) < 8) return true;else return false;
        case 0:return true;
        case 1:return (getvarvalue(c->exp1) == getvarvalue(c->exp2));
        case 2:return (getvarvalue(c->exp1) != getvarvalue(c->exp2));
        default:printf("getcondvalue type error\n");return false;
    }
}

void printvarvalue()
{
    
    int numofvars = numpublicvars + numprivatevars * numprog;
    int i;
    for(i = 0;i < numpublicvars;i++)
    {
        printf("v%d=%d",i,varvalue[i]);
        if(i < 10)
            printf("  ");
        else
            printf(" ");
        
        if(i % 8 == 7)
            printf("\n");
    }
    printf("\n");
    for(i = 0;i < numprog;i++)
    {
        int j;
        for(j = 0;j < numprivatevars;j++)
        {
            printf("p%d.v%d=%d",i,j,varvalue[numpublicvars + i * numprivatevars + j]);
            
            if(i < 10 && j < 10)
                printf("   ");
            else if(i < 10 || j < 10)
                printf("  ");
            else
                printf(" ");
            
            if(j % 8 == 7)
                printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

void printTrace(trace* t,int length)
{
    int i;
    printf("\nprinttrace,length:%d\n",t->steplength);
    for(i = 0;i < length && i < t->steplength;i++)
    {
        printf("executeprogid:%d, type:%d, varvalue:",t->executeprogid[i],t->executenode[i]->type);
        if(t->executenode[i]->next != NULL)
            printf("next type:%d",t->executenode[i]->next->type);
        printf("\n");
        int j;
        //for(j = 0;j < t->numofvar;j++)
        //    printf("v[%d]=%d;",j,t->valueofvar[i][j]);
        printf("\n");
    }
    
}

/*bool checkVarValueChanged(program* prog)
 {
 free(varvalue);
 varvalue = (int*)malloc(numpublicvars + numprivatevars * numprog);
 treenode* next = prog->root;
 treenode* current = NULL;
 
 int i;
 for(i = 0;i < numpublicvars + numprivatevars * numprog;i++)
 varvalue[i] = 0;
 int count = 0,step = 0;
 while(count < 2 && step < 50)
 {
 step++;
 while(next->type == 3)
 next = next->treenode1;
 current = next;
 bool condition;
 //printf("type=%d\n",next->type);
 switch(next->type)
 {
 case 0:    condition = getcondvalue(next->cond1);
 if(condition)
 next = next->treenode1;
 else
 next = next->next;
 break;
 case 1:    if(next->cond1->type == -1)
 {
 count++;
 condition = true;
 }
 else
 condition = getcondvalue(next->cond1);
 
 if(condition)
 {
 if(next->treenode1 != NULL)
 next = next->treenode1;
 }
 else
 next = next->next;
 break;
 case 4:    varvalue[next->index] = getvarvalue(next->exp1);
 next = next->next;
 break;
 case 5: next = next->next;
 break;
 }
 }
 
 if(step < 50 && varvalue[0] == 0 && varvalue[1] == 0)
 return false;
 else
 return true;
 }*/

trace* gettrace(organism* org,int num)                          //???!
{
    
    gtrace->root = org->progs;
    int i,j,k;
    int step = gtrace->steplength;            //???
    //int* stepincs = (int*)malloc(sizeof(int) * numprog);
    
    for(i = 0;i < numprog;i++)
    {
        nextstep[i] = gtrace->root[i]->root;
        currentstep[i] = NULL;
    }
    for(i = 0;i < numprog + 1;i++)
        semavalue[i] = -1;
    for(i = 0;i < gtrace->numofvar;i++)
        varvalue[i] = 0;
    
    
    //printf("22\n");
    for(i = 0;i < step;i++)
    {
        //gtrace->valueofsema[i] = (int*)malloc(sizeof(int) * (numprog + 1));
        bool allprogdone = true;
        for(j = 0;j < numprog;j++)
        {
            if(nextstep[j] != NULL)
            {
                allprogdone = false;
            }
        }
        if(allprogdone)
        {
            gtrace->steplength = i;        //!!
            break;
        }
        
        //printf("23\n");
        int executeprogid;
        
        
        do
        {
            executeprogid = nextrand(numprog);
        }while(nextstep[executeprogid] == NULL);
        
        currentstep[executeprogid] = nextstep[executeprogid];
        gtrace->executeprogid[i] = executeprogid;
        gtrace->executenode[i] = currentstep[executeprogid];
        //printf("24\n");
        bool condition;
        switch(nextstep[executeprogid]->type)
        {
            case 0:    condition = getcondvalue(nextstep[executeprogid]->cond1);
                
                if(condition && nextstep[executeprogid]->treenode_children[0] != NULL)
                    nextstep[executeprogid] = nextstep[executeprogid]->treenode_children[0];
                else
                    nextstep[executeprogid] = nextstep[executeprogid]->next;
                break;
            case 1:    if(nextstep[executeprogid]->cond1->type == -1)
            {
                //if(executeprogid == 1 && num < numofcheck * 0.85 || executeprogid == 0 && (num < numofcheck * 0.7 || num >= numofcheck * 0.85))//num of check
                condition = true;
                //else
                //    condition = false;
                
                //if(condition)
                //    printf("prog:%d,num:%d,true\n",executeprogid,num);
                //else
                //    printf("prog:%d,num:%d,false\n",executeprogid,num);
            }
            else
                condition = getcondvalue(nextstep[executeprogid]->cond1);
                
                if(condition)
                {
                    if(nextstep[executeprogid]->treenode_children[0] != NULL)
                        nextstep[executeprogid] = nextstep[executeprogid]->treenode_children[0 ];
                }
                else
                    nextstep[executeprogid] = nextstep[executeprogid]->next;
                break;
                //case 2:    nextstep[executeprogid] = nextstep[executeprogid]->treenode1;
                //        break;
            case 3://printprog(nextstep[executeprogid],0);
                varvalue[nextstep[executeprogid]->index + 1] = getvarvalue(nextstep[executeprogid]->exp1);
                nextstep[executeprogid] = nextstep[executeprogid]->next;
                break;
            case 4: //printf("wait,sema->index = %d",nextstep[executeprogid]->sema1->index);
                if(getsemavalue(nextstep[executeprogid]->sema1) == -1)        //wait
                {
                    semavalue[nextstep[executeprogid]->sema1->index] = executeprogid;
                    nextstep[executeprogid] = nextstep[executeprogid]->next;
                }
                break;
            case 5: if(getsemavalue(nextstep[executeprogid]->sema1) == executeprogid)//signal
                semavalue[nextstep[executeprogid]->sema1->index] = -1;
                nextstep[executeprogid] = nextstep[executeprogid]->next;
                break;
            case 6:    //if(nextrand(2) == 0)                            //eat
                nextstep[executeprogid] = nextstep[executeprogid]->next;
                break;
            case 7:    //if(nextrand(2) == 0)                            //think
                nextstep[executeprogid] = nextstep[executeprogid]->next;
                break;
        }
        //printf("step:%d,exeprogid:%d\n",i,executeprogid);
        
        //printvarvalue();
        /*gtrace->valueofvar[i] = (int*)malloc(sizeof(int) * gtrace->numofvar);
         for(k = 0;k < gtrace->numofvar;k++)
         {
         gtrace->valueofvar[i][k] = varvalue[k];
         }*/
        
        for(k = 0;k < numprog + 1;k++)
        {
            gtrace->valueofsema[i][k] = semavalue[k];
        }
    }
    
    return gtrace;
}

int getVarindexFromState(State* st,char* name)
{
    //printf("search %s in state\n",name);
    int i;
    for(i = 0;i < st->numvar;i++)
        if(strcmp(st->varname[i],name) == 0)
            break;
    if(i < st->numvar)
        return i;
    else
    {
        printf("ERROR variable name:%s search in state\n",name);
        for(i = 0;i < st->numvar;i++)
            printf("%d:%s\n",i,st->varname[i]);
        exit(-1);
    }
}

int getVarvalueFromState(State* st,char* name)
{
    //printf("xxx search %s in state\n",name);
    int i;
    for(i = 0;i < st->numvar;i++)
        if(strcmp(st->varname[i],name) == 0)
            break;
    if(i < st->numvar)
        return st->varvalue[i];
    else
    {
        printf("ERROR variable name:%s search in state\n",name);
        for(i = 0;i < st->numvar;i++)
            printf("%d:%s\n",i,st->varname[i]);
        exit(-1);
    }
}

void setTraceStates(trace* t)
{
    int eatindex[3],eatingindex[3],sindex[3];
    eatindex[0] = getVarindexFromState(t->states[0],"eat0");
    eatindex[1] = getVarindexFromState(t->states[0],"eat1");
    eatindex[2] = getVarindexFromState(t->states[0],"eat2");
    
    eatingindex[0] = getVarindexFromState(t->states[0],"eating0");
    eatingindex[1] = getVarindexFromState(t->states[0],"eating1");
    eatingindex[2] = getVarindexFromState(t->states[0],"eating2");
    
    sindex[0] = getVarindexFromState(t->states[0],"s0");
    sindex[1] = getVarindexFromState(t->states[0],"s1");
    sindex[2] = getVarindexFromState(t->states[0],"s2");
    
    int i,j;
    int lastnodecs[2];
    lastnodecs[0] = 0;
    lastnodecs[1] = 0;
    
    for(j = 0;j < t->states[0]->numvar;j++)
        t->states[0]->varvalue[j] = 0;
    t->states[0]->varvalue[sindex[0]] = -1;
    t->states[0]->varvalue[sindex[1]] = -1;
    t->states[0]->varvalue[sindex[2]] = -1;
    
    for(i = 0;i < t->steplength;i++)
    {
        for(j = 0;j < t->states[0]->numvar;j++)
            t->states[i + 1]->varvalue[j] = t->states[i]->varvalue[j];
        
        int id = t->executeprogid[i];
        if(t->executenode[i]->type == 7)
        {
            t->states[i + 1]->varvalue[eatingindex[id]] = 1;
            t->states[i + 1]->varvalue[eatindex[id]]++;
        }
        else
            t->states[i + 1]->varvalue[eatingindex[id]] = 0;
        
        t->states[i + 1]->varvalue[sindex[0]] = t->valueofsema[i][0];
        t->states[i + 1]->varvalue[sindex[1]] = t->valueofsema[i][1];
        t->states[i + 1]->varvalue[sindex[2]] = t->valueofsema[i][2];
    }
}

double calculateFitness(organism* prog,Expr** exp,int numexp,double* coef)
{
    int i;
    
    double* result = (double*)malloc(sizeof(double) * numexp);
    for(i = 0;i < numexp;i++)
        result[i] = 0;
    
    int count = 0;
    for(i = 0;i < numofcheck;i++)
    {
        trace* t = gettrace(prog,i);
        
        setTraceStates(t);
        
        int j;
        for(j = 0;j < numexp;j++)
        {
            double value = getExprValue(exp[j],t,0);
            result[j] += value;
        }
        
        
    }
    
    for(i = 0;i < numexp;i++)
    {
        result[i] = result[i] / (double)numofcheck;
    }
    
    double fitness = 0;
    for(i = 0;i < numexp;i++)
    {
        fitness += coef[i] * result[i];
        prog->progs[0]->propertyfit[i] = result[i];
    }
    
    
    return (fitness * 100);
}

void initTraceGlobalVar(int steplength)
{
    gtrace = (trace*)malloc(sizeof(trace));
    gtrace->steplength = steplength;
    gtrace->numofprog = numprog;
    gtrace->numofvar = numpublicvars + numprivatevars * numprog;
    
    varvalue = (int*)malloc(sizeof(int) * gtrace->numofvar);
    nextstep = (treenode**)malloc(sizeof(treenode*) * gtrace->numofprog);
    currentstep = (treenode**)malloc(sizeof(treenode*) * gtrace->numofprog);
    semavalue = (int*)malloc(sizeof(int) * (numprog + 1));
    
    //gtrace->valueofvar = (int**)malloc(sizeof(int*) * steplength);
    gtrace->executeprogid = (int*)malloc(sizeof(int) * steplength);
    gtrace->valueofsema = (int**)malloc(sizeof(int*) * steplength);
    if(gtrace->executeprogid == NULL)
    {
        printf("failed allocaating!\n");
        perror("do we have an error\n");
    }
    gtrace->executenode = (treenode**)malloc(sizeof(treenode*) * steplength);
    gtrace->satisfied = nextrand(2) == 0 ? true : false;
    
    gtrace->states = (State**)malloc(sizeof(State*) * (steplength + 1));
    int i;
    
    for(i = 0;i < steplength;i++)
        gtrace->valueofsema[i] = (int*)malloc(sizeof(int) * (numprog + 1));
    
    for(i = 0;i < steplength + 1;i++)
    {
        gtrace->states[i] = (State*)malloc(sizeof(State));
        gtrace->states[i]->numvar = 9;
        gtrace->states[i]->varvalue = (int*)malloc(sizeof(int) * gtrace->states[i]->numvar);
        gtrace->states[i]->varname = (char**)malloc(sizeof(char*) * gtrace->states[i]->numvar);
        gtrace->states[i]->varname[0] = "eat0";
        gtrace->states[i]->varname[1] = "eat1";
        gtrace->states[i]->varname[2] = "eat2";
        gtrace->states[i]->varname[3] = "eating0";
        gtrace->states[i]->varname[4] = "eating1";
        gtrace->states[i]->varname[5] = "eating2";
        gtrace->states[i]->varname[6] = "s0";
        gtrace->states[i]->varname[7] = "s1";
        gtrace->states[i]->varname[8] = "s2";
    }
}


//treerank.c

void addNode(treeRank* tr,treenode* node)
{
    if(node == NULL)
        return;
    
    if(tr->numcandidate == tr->maxnumcandidate)
    {
        tr->maxnumcandidate *= 2;
        mutationNode** new = (mutationNode**)malloc(sizeof(mutationNode*) * tr->maxnumcandidate);
        int i;
        for(i = 0;i < tr->numcandidate;i++)
        {
            new[i] = tr->candidate[i];
        }
        free(tr->candidate);
        tr->candidate = new;
    }
    
    mutationNode* p = (mutationNode*)malloc(sizeof(mutationNode));
    p->node = node;
    
    tr->ranksum += node->badexamples;
    p->ranksum = tr->ranksum;
    tr->candidate[tr->numcandidate] = p;
    tr->numcandidate++;
}

/*
void searchNode(treenode* root,treeRank* tr,int type,int maxdepth)
{
    
    if(root == NULL)
        return;
    if(root->fixed == 0)
    {
        if(type == 1)
        {
            addNode(tr,root);
        }
        else if(type == 2)        //??
        {
            if(root->depth + root->height < maxdepth + 1)
                addNode(tr,root);
            else if(root->depth == 2 && root->numofstatements < 8)
                addNode(tr,root);
            else if(root->depth == maxdepth && root->numofstatements < 4)
            {
                addNode(tr,root);
            }
        }
        else if(type == 3)
        {
            if(satisfyMutationReduction(root) == 1)
                addNode(tr,root);
        }
        else if(type == 4)
        {
            if(root->type == 1)
            {
                if(root->treenode1 != NULL && root->treenode1->fixed == 0 || root->treenode1 == NULL)    //=NULL    add child to empty while
                    addNode(tr,root);
            }
        }
        else if(type == 5)
        {
            if(root->depth + 2 <= maxdepth)            //!!
                addNode(tr,root);
        }
    }
    searchNode(root->treenode1,tr,type,maxdepth);
    searchNode(root->treenode2,tr,type,maxdepth);
}
*/


mutationNode* chooseNode(treeRank* tr)
{
    if(tr->ranksum == 0)
        return NULL;
    //printf("start tr->ranksum:%d\n",tr->ranksum);
    int p = nextrand(tr->ranksum);
    int i;
    mutationNode* result;
    for(i = 0;i < tr->numcandidate;i++)
    {//printf("i:%d\n",i);
        result = tr->candidate[i];
        if(result == NULL)printf("result NULL!!\n");
        if(result->ranksum > p)
            break;
    }
    //printf("end\n");
    for(i = 0;i < tr->numcandidate;i++)
        if(tr->candidate[i] != result)
            free(tr->candidate[i]);
    free(tr);
    
    return result;
}

Expr* createExpr(ExprType t,UnOp u,BinOp b,int bound,char* n,int v,Expr* child,Expr* left,Expr* right)
{
    Expr* result = (Expr*)malloc(sizeof(Expr));
    result->type = t;
    result->uop = u;
    result->bop = b;
    
    result->stepbound = bound;
    //strcpy(result->name,n);
    result->name = n;
    result->value = v;
    
    result->child = child;
    result->left = left;
    result->right = right;
    
    return result;
}
Expr* createExprConstant(int value){
    return createExpr(Constant,0,0,0,NULL,value,NULL,NULL,NULL);
}
Expr* createExprVar(char* name){
    return createExpr(Variable,0,0,0,name,0,NULL,NULL,NULL);
}
Expr* createExprExp1(UnOp op,Expr* child){
    return createExpr(Expr1,op,0,0,NULL,0,child,NULL,NULL);
}
Expr* createExprStepb(UnOp op,Expr* child,int stepbound){
    return createExpr(StepBoundExpr1,op,0,stepbound,NULL,0,child,NULL,NULL);
}
Expr* createExprExpr2(BinOp op,Expr* left,Expr* right){
    return createExpr(Expr2,0,op,0,NULL,0,NULL,left,right);
}

BinOp getBinOp(char* buf,int index)
{
    BinOp op;
    switch(buf[index])
    {
        case '-':if(buf[index + 1] == '>')
            op = Imp;
        else
            op = Min;
            break;
        case '&':op = And;break;
        case '|':op = Or;break;
        case '=':op = Eq;break;
        case '!':if(buf[index + 1] == '=')
        {
            op = Neq;
            break;
        }
        else
        {
            printf("Error:property BinOp !\n");
            exit(-1);
        }
        case '<':if(buf[index + 1] == '=')
            op = Le;
        else
            op = Lt;
            break;
        case '>':if(buf[index + 1] == '=')
            op = Ge;
        else
            op = Gt;
            break;
        case '+':op = Add;break;
        case '*':op = Mul;break;
        default:printf("Error:property BinOp\n");
            exit(-1);
    }
    return op;
}

int getBinOpLength(BinOp op)
{
    if(op == Neq || op == Le || op == Ge || op == Imp)
        return 2;
    else
        return 1;
}

Expr* generateStepExpr(char* buf, int leftindex, int rightindex)
{    //printf("%s,%d,%d\n",buf,leftindex,rightindex);
    if(buf[leftindex] == '(')
    {
        int index = leftindex,count = 0;
        while(index <= rightindex)
        {
            if(buf[index] == '(')
                count++;
            else if(buf[index] == ')')
                count--;
            if(count == 0)
                break;
            index++;
        }
        if(count != 0)
        {
            printf("Error:property parentheses not match!\n");
            exit(-1);
        }
        
        if(index == rightindex)
            return generateStepExpr(buf, leftindex + 1, rightindex - 1);
        else
        {
            Expr* left = generateStepExpr(buf, leftindex + 1, index - 1);
            BinOp op = getBinOp(buf,index + 1);
            Expr* right = generateStepExpr(buf, index + 1 + getBinOpLength(op),rightindex);
            return createExprExpr2(op,left,right);
        }
    }
    
    if(buf[leftindex] == 'G' || buf[leftindex] == 'F' || buf[leftindex] == 'X')
    {
        UnOp op;
        if(buf[leftindex] == 'G')
            op = Globally;
        else if(buf[leftindex] == 'F')
            op = Future;
        else
            op = Next;
        int index = leftindex + 4;
        int stepbound = 0;
        while(index <= rightindex && buf[index] <= '9' && buf[index] >= '0')
        {
            stepbound *= 10;
            stepbound += buf[index++] - '0';
        }
        
        Expr* child = generateStepExpr(buf,index,rightindex);
        Expr* result = createExprStepb(op,child,stepbound);
        
        return result;
    }
    else if(buf[leftindex] <= '9' && buf[leftindex] >= '0')
    {
        int value = 0;
        while(leftindex <= rightindex && buf[leftindex] <= '9' && buf[leftindex] >= '0')
        {
            value *= 10;
            value += buf[leftindex++] - '0';
        }
        return createExprConstant(value);
        
    }
    else if(buf[leftindex] <= 'z' && buf[leftindex] >= 'a')
    {
        int index = leftindex;
        while(index <= rightindex && (buf[index] <= '9' && buf[index] >= '0' || buf[index] <= 'z' && buf[index] >= 'a'))
            index++;
        char* name = (char*)malloc(sizeof(char) * (index - leftindex + 1));
        strncpy(name,&(buf[leftindex]),index - leftindex);
        name[index - leftindex] = '\0';
        Expr* left = createExprVar(name);
        if(index >= rightindex)
            return left;
        else
        {
            BinOp op = getBinOp(buf,index);
            Expr* right = generateStepExpr(buf,index + getBinOpLength(op),rightindex);
            return createExprExpr2(op,left,right);
        }
        /*while(index < rightindex)
         {
         BinOp op = getBinOp(buf,index);
         Expr* right = generateStepExpr(buf,index + getBinOpLength(op),rightindex);
         createExprExpr2(op,left,right);
         }*/
    }
    else
    {
        printf("Error:property!\n");
        exit(-1);
    }
}

int getStepBound(Expr* exp)
{
    if(exp == NULL)
        return 0;
    
    int left,right;
    switch(exp->type)
    {
        case Constant:return 0;
        case Variable:return 0;
        case Expr1:return getStepBound(exp->child);
        case StepBoundExpr1:return (exp->stepbound + getStepBound(exp->child));
        case Expr2:    left = getStepBound(exp->left);
            right = getStepBound(exp->right);
            return (left>right?left:right);
    }
}

int getExprValue(Expr* exp,trace* t,int currentStep)
{
    //printf("get expr value:");
    //printExpr(exp);
    //printf("\n");
    if(exp->type == Constant)
        return exp->value;
    else if(exp->type == Variable)
        return getVarvalueFromState(t->states[currentStep],exp->name);
    else if(exp->type == Expr1)
    {
        if(exp->uop == Neg)
        {
            return (-1 * getExprValue(exp->child,t,currentStep));
        }
        else if(exp->uop == Not)
        {
            if(getExprValue(exp->child,t,currentStep) == 0)
                return 1;
            else
                return 0;
        }
        else
        {
            printf("ERROR:Expr1 uop type invalid!\n");
            exit(-1);
        }
    }
    else if(exp->type == StepBoundExpr1)
    {
        
        if(exp->uop == Future)
        {
            int i;
            for(i = currentStep;i <= currentStep + exp->stepbound;i++)            //important
                if(getExprValue(exp->child,t,i) == 1)
                    return 1;
            return 0;
        }
        else if(exp->uop == Globally)
        {
            int i;
            for(i = currentStep;i <= currentStep + exp->stepbound;i++)            //important
                if(getExprValue(exp->child,t,i) == 0)
                    return 0;
            return 1;
        }
        else if(exp->uop == Next)
        {
            if(getExprValue(exp->child,t,currentStep + exp->stepbound) == 1)
                return 1;
            else
                return 0;
        }
        else
        {
            printf("ERROR:StepBoundExpr1 uop type invalid!\n");
            exit(-1);
        }
    }
    else if(exp->type == Expr2)
    {
        int left = getExprValue(exp->left,t,currentStep);
        int right;
        switch(exp->bop)
        {
            case Imp:    if(left > 0 && getExprValue(exp->right,t,currentStep) == 0)
                return  0;
            else
                return 1;
            case And:    if(left > 0 && getExprValue(exp->right,t,currentStep) > 0)
                return 1;
            else
                return 0;
            case Or:    if(left > 0 || getExprValue(exp->right,t,currentStep) > 0)
                return  1;
            else
                return 0;
            case Eq:    if(left == getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Neq:    if(left != getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Lt:    if(left < getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Le:    if(left <= getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Gt:    if(left > getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Ge:    if(left >= getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Add:    return (left + getExprValue(exp->right,t,currentStep));
            case Min:    return (left - getExprValue(exp->right,t,currentStep));
            case Mul:    return (left * getExprValue(exp->right,t,currentStep));
            case Div:    right = getExprValue(exp->right,t,currentStep);
                if(right != 0)
                    return (left / right);
                else
                    return 0;
            default:    printf("ERROR:getExprValue Expr2 BinOp type invalid!\n");
        }
    }
    else
    {
        printf("Error:Expr type invalid!\n");
        exit(-1);
    }
}

double* set_coef(int numofrequirements)
{
    double* coef = (double*)malloc(sizeof(double) * numofrequirements);
    for(int i = 0;i < numofrequirements;i++)
        coef[i] = 1.0 / (double)numofrequirements;
    for(int i = 0;i < numofrequirements;i++)
        printf("%f", coef[i]);
    // coef[0] = 0.6;coef[1] = 0.2;coef[2] = 0;coef[3] = 0;
    // coef[4] = 0.2;coef[5] = -0.1;coef[6] = -0.2;
    // printf("x1\n");
    return coef;
}

Expr** set_requirments(int numofrequirements)
{
    int i,j,k;
    
    Expr** requirements = (Expr**)malloc(sizeof(Expr*) * numofrequirements);
    
    for(i = 0;i < numofrequirements;i++)
    {
        FILE* fp = NULL;
        char buf[255];
        switch(i)
        {
            case 0:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/A0.bltl","r");break;
            case 1:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/A1.bltl","r");break;
            case 2:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/A2.bltl","r");break;
            case 3:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/B0.bltl","r");break;
            case 4:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/B1.bltl","r");break;
            case 5:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/B2.bltl","r");break;
            case 6:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/C0.bltl","r");break;
            case 7:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/C1.bltl","r");break;
            case 8:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/C2.bltl","r");break;
            case 9:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/D0.bltl","r");break;
            case 10:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/D1.bltl","r");break;
            case 11:fp = fopen("/Users/zhuang/worksPPO/philoserpher/philoserpher/property/D2.bltl","r");break;
                
        }
        
        fgets(buf, 255, fp);
        
        fclose(fp);
        //printf("%s\n%d\n",buf,strlen(buf));
        int index = 0;
        j = 0;
        while(buf[j] != '\0')
        {
            if(buf[j] != ' ' && buf[j] != '\n')
            {
                buf[index] = buf[j];
                index++;
            }
            j++;
        }
        buf[index] = '\0';
        //printf("%s\n%d\n",buf,strlen(buf));
        
        Expr* e = generateStepExpr(buf,0,strlen(buf) - 1);
        //printf("xy,%d,%d\n",i,sizeof(requirements));
        requirements[i] = e;
        //printExpr(e);
        //printf("\n");
    }
    int maxstepbound = 0;
    for(i = 0;i < numofrequirements;i++)
    {
        int step = getStepBound(requirements[i]);
        if(step > maxstepbound)
            maxstepbound = step;
    }
    initTraceGlobalVar(maxstepbound + 1);
    
    
    
    return requirements;
}

program* initProg(Expr** requirements ,int numofrequirements)
{
    double* coef = (double*)malloc(sizeof(double) * numofrequirements);
    for(int i = 0;i < numofrequirements;i++)
        coef[i] = 1.0 / (double)numofrequirements;
    
    int numofcandidate = 1;
    program** candidate = genInitTemplate(numofcandidate);
    for(int i = 0;i < numofcandidate;i++)
    {
        organism* org = genOrganism(candidate[i]);
        double candidatefit = calculateFitness(org,requirements,numofrequirements,coef);
        candidate[i]->fitness = candidatefit;
        freeAll(org,NULL,NULL,NULL,NULL,1);
    }
    //printprog(candidate[0]->root, 0 , candidate[0]);
    //printf("%f\n", candidate[0]->fitness);
    // printAst(candidate[0]);
    //int* vector = genVector(candidate[0]);
    //for (int i = 0 ; i < 40; i++) {
    //    printf("%d", vector[i]);
    //}
    return candidate[0];
}

int NUM_CONDITION_LEFT = 4;
int NUM_CONDITION_RIGHT = 6;
int NUM_ASSIGNMENT_LEFT = 2;
int NUM_ASSIGNMENT_RIGHT = 2;
int NUM_TREENODE = 40;

void setTreenodeNum(treenode* root, program* prog, int number)
{
    //printf("parent type :%d",prog->parent->type)
    root->number = number;
    if(root->depth == 1)
    {
        if(root->treenode_children[0] != NULL)
        {
            setTreenodeNum(root->treenode_children[0], prog,number + 1);
        }
        int i = 0;
        while(root->treenode_children[i + 1] != NULL)
        {
            setTreenodeNum(root->treenode_children[i + 1], prog, root->treenode_children[i]->number + 5);
            i++;
        }
    }
    else if(root->depth == 2)
    {
        if(root->treenode_children[0] != NULL)
        {
            setTreenodeNum(root->treenode_children[0], prog,number + 1);
        }
        int i = 0;
        while(root->treenode_children[i + 1] != NULL)
        {
            setTreenodeNum(root->treenode_children[i + 1], prog, root->treenode_children[i]->number + 1);
            i++;
        }
    }
}

int getConditionId(cond* c)        //start from 1
{
    if(c == NULL)
        return 0;
    int result = 0;
    if(c->type <= 2)
    {
        result += (c->exp1->index + 4) * NUM_CONDITION_RIGHT;
        if(c->exp2->type == 0)
            result += c->exp2->index + 1;
        else
            result += c->exp2->index + 7;
        if(c->type == 2)
            result += NUM_CONDITION_LEFT * NUM_CONDITION_RIGHT;
    }
    return result;
}

int getTreenodeId(treenode* node)
{
    if(node == NULL)
        return 0;
    if(node->type == 0)
        return getConditionId(node->cond1);
    else if(node->type == 1)
    {
        int temp = NUM_CONDITION_LEFT * 2 * NUM_CONDITION_RIGHT;
        return (getConditionId(node->cond1) + temp);
    }
    else if(node->type == 3)
    {
        int result = 0;
        int temp = NUM_CONDITION_LEFT * 2 * NUM_CONDITION_RIGHT;
        result += temp * 2;
        result += (node->index + 2) * NUM_ASSIGNMENT_RIGHT;
        result += node->exp1->index + 1;
        return result;
    }
    else if(node->type == 4)
    {
        int result = 0;
        int temp = NUM_CONDITION_LEFT * 2 * NUM_CONDITION_RIGHT;
        result += temp * 2;
        result += NUM_ASSIGNMENT_LEFT * NUM_ASSIGNMENT_RIGHT;
        result += node->sema1->index + 4;
        return result;
    }
    else if(node->type == 5)
    {
        int result = 0;
        int temp = NUM_CONDITION_LEFT * 2 * NUM_CONDITION_RIGHT;
        result += temp * 2;
        result += NUM_ASSIGNMENT_LEFT * NUM_ASSIGNMENT_RIGHT;
        result += 3;
        result += node->sema1->index + 4;
        return result;
    }
    else if(node->type == 6)
    {
        int result = 0;
        int temp = NUM_CONDITION_LEFT * 2 * NUM_CONDITION_RIGHT;
        result += temp * 2;
        result += NUM_ASSIGNMENT_LEFT * NUM_ASSIGNMENT_RIGHT;
        result += 3;
        result += 3;
        return (result + 1);
    }
    else
    {
        int result = 0;
        int temp = NUM_CONDITION_LEFT * 2 * NUM_CONDITION_RIGHT;    //if
        result += temp * 2;                                            //if+while
        result += NUM_ASSIGNMENT_LEFT * NUM_ASSIGNMENT_RIGHT;        //assignment
        result += 3;                                                //wait
        result += 3;                                                //signal
        result += 1;                                                //think
        return (result + 1);
    }
}

void genVectorTreenode(treenode* node, int* id)
{
    if(node == NULL)
        return;
    if(node->number > NUM_TREENODE)
    {
        printf("%d node->number: ", node->number);
        printf("Error:treenode number greater than 40!\n");
        return;
    }
    if(node->type != 2)
        if(node->number != 0)
            id[node->number - 1] = getTreenodeId(node);

    //printf("numofstatements %d\n", node->numofstatements);
    for (int i = 0; i < node->numofstatements; i++)
    {

        genVectorTreenode(node->treenode_children[i],id);
    }

}

int* genVector(program* prog)
{
    int *vector = (int*)malloc(sizeof(int) * NUM_TREENODE);
    int i = 0;
    for(i = 0;i < NUM_TREENODE;i++)
        vector[i] = 0;

    genVectorTreenode(prog->root,vector);
    /*
    for (int k = 0; k<40; k++) {
        printf("\n%d", vector[k]);
    }
    */
    return vector;
}



exp_* genexp_new(int varindex)
{
    exp_* result = NULL;
    int vartype = 1;
    if (varindex >= 0) {
        vartype = 0;
    }
    result = createExp(vartype, varindex);
    return result;
}

cond* gencond_new(int condtype, int expindex1, int expindex2)    //type==3:true,==,!=    type==5:true,==,!=,&&,||
{
    cond* result = NULL;
    result = createCond(condtype,genexp_new(expindex1),genexp_new(expindex2));
    return result;
}

//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN    5:critical section
//treenode* createTreenode(int t,int ind,cond* c,treenode* t1,treenode* t2,treenode* f,exp* e);
treenode* genprog_new(int commandtype, int condi, int expindex1, int expindex2, int sema)
{
    int varindex = 0;
    treenode* result = NULL;
    switch(commandtype)
    {
        case 0: result = createTreenode(0,0,gencond_new(condi, expindex1, expindex2),NULL,NULL,NULL);    //0:TRUE
            
            // result->treenode1 = genprog(depth + 1,prog);
            break;
        case 1: result = createTreenode(1,0,gencond_new(condi, expindex1, expindex2),NULL,NULL,NULL);
            
            break;
        case 3: varindex = expindex1 - 2;    //??
            result = createTreenode(3,varindex,NULL,NULL,NULL,genexp_new(expindex2));
            break;
        case 4: result = createTreenode(4,0,NULL,NULL,NULL,NULL);
            result->sema1 = gensema_new(sema);
            break;
        case 5:    result = createTreenode(5,0,NULL,NULL,NULL,NULL);
            result->sema1 = gensema_new(sema);
            break;
            
    }
    return result;
}

int findeat(program* newprog)
{
    int result = 0;
    program* prog = copyProgram(newprog);
    treenode* root_temp = prog->root;
    if (root_temp->type == 1)
    {
        for (int i = 0; i < root_temp->numofstatements; i++)
        {
            while (root_temp->treenode_children[i]->type == 7)
            {
                result = i;
                break;
            }
        }
    }
    return result;
}

treenode* gennewtreenode(int mutationtype2, program* prog)
{
    // printf("gennewtreenod mutationtype2 %d \n", mutationtype2);
    treenode* newnode = NULL;
    int treenodetype = 0;
    int mutationtype = mutationtype2 % 66;
    if(mutationtype < 66)
    {
        int cut1 = mutationtype / 56;
        if(cut1 == 0)                  //if while
        {
            int treenodetype = mutationtype / 28;
            int ifstate = mutationtype % 28;
            int condtype = ifstate / 14 + 1;        // 1: eq 2: neq
            int condstate = ifstate % 14;
            int exp_1;
            int exp_2;
            if(condstate < 3)
            {
                exp_1 = -4;
                exp_2 = condstate - 3;
            }
            else if(condstate < 5)
            {
                exp_1 = -3;
                if (condstate == 3)
                    exp_2 = -2;
                else
                    exp_2 = -1;
                
            }
            else if(condstate < 6)
            {
                exp_1 = -2;
                exp_2 = -1;
            }
            else if (condstate < 14)
            {
                exp_2 = (condstate - 6) / 4;
                exp_1 = (condstate - 6) % 4 - 4;
            }

            newnode = genprog_new(treenodetype, condtype, exp_1, exp_2, 0);
            
            // freeAll(0, 0, newnode, 0, 0, 0);
            
        }
        else
        {
            int temp = mutationtype - 56;
            if (temp < 4)                            // assign
            {
                treenodetype = 3;
                if(temp < 0)
                    printf("mutation error");
                int assign_index = temp / 2;
                int assign_exp = temp % 2;
                newnode = genprog_new(treenodetype, 0, assign_index, assign_exp, 0);
                //printf("\ntreenode->index:%d", newnode->index);
                //printf("\ntreenode->exp1->index :%d", newnode->exp1->index);
                
            }
            else                                   // wait signal
            {
                treenodetype = (temp - 4) / 3 + 4;
                int sema = (temp - 4) % 3;
                newnode = genprog_new(treenodetype, 0, 0, 0, sema);
                
                
            }
            
        }
    }
    //setNumOfStatements(newnode);
    //printprog(newnode, 0, prog);
    //printf("!!!\n");
    return newnode;
}

NUMBERDEPTH2BF = 66;
NUMBERDEPTH2AF = 66;
NUMBERDEPTH3BF = 10;
NUMBERDEPTH3AF = 10;

int* legalAction(program* parent)
{
    int* action = (int*)malloc(sizeof(int)*3);
    for (int i = 0; i<3; i++)
    {
        action[i] = 0;
    }
    // memset(action, 0, sizeof(int)*3);
    int index = 1;
    //printprog(parent->root, 0, parent);
    program* newprog = copyProgram(parent);
    //printprog(parent->root, 0, parent);
    int eatIndex = findeat(newprog);
    treenode* root = newprog->root;
    int addKinds = 1;
    int addBF = 0;
    int addAF = 0;
    if(root->treenode_children[eatIndex - 1]->type == 0 || root->treenode_children[eatIndex - 1]->type == 1 || root->treenode_children[root->numofstatements-1]->type == 0 || root->treenode_children[root->numofstatements-1]->type == 1)
    {
        if(((root->treenode_children[eatIndex - 1]->type == 0 || root->treenode_children[eatIndex - 1]->type == 1) && root->treenode_children[eatIndex - 1]->numofstatements == 0))
                addBF = 1;
        if(((root->treenode_children[root->numofstatements-1]->type == 0 || root->treenode_children[root->numofstatements-1]->type == 1) && root->treenode_children[root->numofstatements-1]->numofstatements == 0))
                addAF = 1;

        if ((root->treenode_children[eatIndex - 1]->type == 0 || root->treenode_children[eatIndex - 1]->type == 1) && root->treenode_children[eatIndex - 1]->numofstatements < 4 && addAF != 1)
        {
            action[index] = NUMBERDEPTH3BF;
            index ++;
        }
        else
        {
            action[index] = 0;
            index ++;
        }
        
        if ((root->treenode_children[root->numofstatements-1]->type == 0 || root->treenode_children[root->numofstatements-1]->type == 1) && root->treenode_children[root->numofstatements-1]->numofstatements < 4 && addKinds != 0 && addBF != 1)
        {

            action[index] = NUMBERDEPTH3AF;
            index ++;
        }
        else
        {
            action[index] = 0;
            index ++;
        }
    }
    if(addAF != 1 && addBF != 1)
    {
        if (root->numofstatements < 8 )
        {
            action[0] = NUMBERDEPTH2AF + NUMBERDEPTH2BF;
            //index ++;
        }
        else
        {
            action[0] = 0;
            //index ++;
        }
    }
    //for(int k = 0; k<3;k++)
    //    printf("%d\n", action[k]);
    //printf("%p\n", action);
    return action;
}

program* mutation_new(program* parent, int mutationtype, Expr** requirements, int numofrequirements)
{
    double* coef = (double*)malloc(sizeof(double) * numofrequirements);
    for(int i = 0;i < numofrequirements;i++)
        coef[i] = 1.0 / (double)numofrequirements;

    program* newprog = copyProgram(parent);
    newprog->checkedBySpin = 0;
    int eatIndex = findeat(newprog);
    int max = mutationtype / 132;
    int miniIndex = mutationtype % 132;
    treenode* root = newprog->root;
    treenode* chnod = root->treenode_children[eatIndex - 1];
    
    
    if(max == 0)
    {
        int BFAF = mutationtype / 66;
        int afIndex = mutationtype % 66;
        int changeindex = root->numofstatements;
        if (BFAF == 0)
        {
            while (eatIndex < changeindex)
            {
                root->treenode_children[changeindex] = root->treenode_children[changeindex - 1];
                changeindex--;
            }
            //printf("BFAF == 0 mutation_new mutationtype %d\n", mutationtype);
            root->treenode_children[eatIndex] = gennewtreenode(mutationtype, newprog);
        }
        else
        {
            //printf("changeindex %d\n", changeindex);
            //printf("BFAF == 1 mutation_new mutationtype %d\n", afIndex);
            root->treenode_children[changeindex] = gennewtreenode(afIndex, newprog);
        }
    }
    else
    {
        int BFAF = miniIndex / 10;
        int AFmini = miniIndex % 10;
        if (BFAF == 0)
        {
            //printf("max == 1 BFAF == 0 %d\n", miniIndex);
            chnod->treenode_children[chnod->numofstatements] = gennewtreenode(miniIndex + 56, newprog);
        }
        else
        {
            //printf("max == 1 BFAF == 1 %d\n", AFmini);
            root->treenode_children[root->numofstatements - 1]->treenode_children[root->treenode_children[root->numofstatements - 1]->numofstatements] = gennewtreenode(AFmini + 56, newprog);
        }
    }
    newprog->root = root;
    setAll(newprog);
    //printprog(newprog->root, 0, newprog);
    organism* org = genOrganism(newprog);
    newprog->fitness = calculateFitness(org, requirements, numofrequirements, coef);
    //printf("fitness : %f\n", newprog->fitness);
    return newprog;
}


int spin_(program* candidate)
{
    int numofspin = 0;
    int numofsolution = 0;
    int right = 0;
    int reward = 0;
    printf("candidate->fitness = %f", candidate->fitness);

    if(candidate->fitness > 98)
    {
        //candidate[j]->checkedBySpin = 1;
        organism* org = genOrganism(candidate);
        printf("2xx\n");
        FILE* f;
        char filename[60] = "./output/mutex.pml";
        // strcat(filename,argv[1]);
        // strcat(filename,".pml");
        if(f = fopen(filename,"w"))
        {
            orgToPml(org,f);
        }
        //sleep(2);
        fclose(f);
        printf("1xx\n");
        char command[50] = "spin -a ";
        strcat(command,filename);
        strcat(command," > useless");

        printf("xx\n");
        system(command);
        system("gcc -DMEMLIM=1024 -O2 -DXUSAFE -w -o pan pan.c");
        printf("xx3\n");
        numofspin++;
        system("./pan -m10000 -a -f -N e1 > pan1.out");
        printf("xx4\n");
        int r1 = system("grep -q -e \"errors: 0\" pan1.out");
        if(r1 == 0)
        {
            printf("e1");
            reward += 5;
        }

        //if(r1 == 0)    candidate[j]->fitness += 10;

        numofspin++;
        system("./pan -m10000 -a -f -N e2 > pan2.out");
        int r2 = system("grep -q -e \"errors: 0\" pan2.out");
        if(r2 == 0)
        {
            printf("e2");
            reward += 5;
        }
        //if(r2 == 0)    candidate[j]->fitness += 5;

        numofspin++;
        system("./pan -m10000 -a -f -N e3 > pan3.out");
        int r3 = system("grep -q -e \"errors: 0\" pan3.out");
        if(r3 == 0)
        {
            printf("e3");
            reward += 5;
        }
        //if(r3 == 0)    candidate[j]->fitness += 5;

        numofspin++;
        system("./pan -m10000 -a -f -N e4 > pan4.out");
        int r4 = system("grep -q -e \"errors: 0\" pan4.out");
        if(r4 == 0)
        {
            printf("e4");
            reward += 5;
        }
        //if(r3 == 0)    candidate[j]->fitness += 5;

        numofspin++;
        system("./pan -m10000 -a -f -N e5 > pan5.out");
        printf("xx5\n");
        int r5 = system("grep -q -e \"errors: 0\" pan5.out");
        printf("xx\n");
        if(r5 == 0)
        {
            printf("e5");
            reward += 5;
        }
        //if(r3 == 0)    candidate[j]->fitness += 5;

        numofspin++;
        system("./pan -m10000 -a -f -N e6 > pan6.out");
        printf("xx3\n");
        int r6 = system("grep -q -e \"errors: 0\" pan6.out");
        if(r6 == 0)
        {
            printf("e6");
            reward += 5;
        }
        //if(r3 == 0)    candidate[j]->fitness += 5;

        if(reward == 30)
        {
            printf("Correct Solution%d\n",numofsolution++);
            printprog(candidate->root,0,candidate);
            printprog(org->progs[1]->root,0,org->progs[1]);
            printprog(org->progs[2]->root,0,org->progs[2]);
            printf("\n");
            right = 1;
            //printprog(candidate[j]->root,0,candidate[j]);
        }
        freeAll(org,NULL,NULL,NULL,NULL,1);
    }
    return reward;
}



