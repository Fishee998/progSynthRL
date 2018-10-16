# progSynthRL
我的系统是macOS High Sierra 10.13.5
python 2.7

1 安装swig
 # brew install swig
 
2 用swig包装c函数使之可以被python调用 
 # make

3 安装gym
 # pip install gym
  
4 将gym内的cartpole.py替换成这边的cartpole.py
  /Library/Python/2.7/site-packages/gym/envs/classic_control/cartpole.py  //取决于python的安装位置
  
5 python MyDQN.py
tips： 1 如果没有修改example.c程序就不需要重新make了
       2 Makefile里面的路径根据python实际的路径修改
       3 调用c函数出问题多半是路径文件路径问题 
       
       
