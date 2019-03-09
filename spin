int spin_(program* candidate)
{
    int numofspin = 0;
    int numofsolution = 0;
    int right = 0;
    int reward = 0;
    printf("candidate->fitness = %f", candidate->fitness);

    if(candidate->fitness > 40)
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
            printf("program0\n");
            printprog(org->progs[0]->root,0,org->progs[0]);
            printf("program1\n");
            printprog(org->progs[1]->root,0,org->progs[1]);
            printf("program2\n");
            printprog(org->progs[2]->root,0,org->progs[2]);
            printf("\n");
            right = 1;
            //printprog(candidate[j]->root,0,candidate[j]);
        }
        freeAll(org,NULL,NULL,NULL,NULL,1);
    }
    return reward;
}
