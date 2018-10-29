# progSynthRL

MacOS High Sierra 10.13.5
Linux 17.10
python版本2.7


1 安装Swig 

3 用swig包装c文件使之可以被python调用

修改MAKEFILE中   -I的值 -I的值可以通过以下命令获得 

    $(pkg-config --cflags --libs python2）
  
     make 
   
4   运行

     python run_this.py 
  
tips： 

  1 reward的设计说明在reward.txt
  
  2 探索率的衰减变量为e_greedy_increment （RL_brain）
  
  3 输出由两个网络，分别是35维的action1（选择ast的哪一个节点），50维的action2（结点的变换操作）
  
  4 选择action时用了人工干预，主要是过滤掉不符合规则的变换方式，限制条件action1不选空结点，action2选择结点可执行的操作。
