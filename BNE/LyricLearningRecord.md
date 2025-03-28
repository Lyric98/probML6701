## Git init
先在github remote建repo
```
rename main -> master
git init
git remote add origin git@github.com:Lyric98/xxxxxxxx.git
git remote -v (检查是否连上)
git pull origin master
git branch -a
```
git 撤销commit： https://www.cnblogs.com/lfxiao/p/9378763.html 
```
git reset --soft HEAD^ #HEAD^的意思是上一个版本，也可以写成HEAD~1
                      #如果你进行了2次commit，想都撤回，可以使用HEAD~2
git log 查询commit
```

## 环境
conda create -n your_env_name python=x.x Python创建虚拟环境

conda activate RL

conda deactivate (退出该环境)

conda env list 查看已有环境list

## vscode 打开项目
comment shift P 输入 “interpreter” 选中所属环境

jupyter notebook 转换成 python script
```
jupyter nbconvert --to script
```
vscode 选中多行缩进

缩进： ```tab```
取消缩进： ```shift + tab```


## Linux 相关
vim 光标移动到行首、行尾 (在enter 编辑 “i” 之前操作)
```
行首 0 (or ^)
行尾 $ 
```
删除行
```
按Esc键进入正常模式。
将光标放在要删除的行上。
键入dd以删除该行。
```


Linux下删除指定文件夹下指定后缀名的文件: 命令有点危险，可以先执行前半段，看看是不是你要删除的文件, 然后再整条执行

```
find . -name "*.out"  
find . -name "*.out"| xargs rm
```
FASRC cancel job
```
scancel --name=train.sh
```


## 引用同地址的文件出现 module not found error
```
pip install -e .
(fsvi) liyanran@liyanrandeMBP function-space-variational-inference-yanran % pip install -e .
Obtaining file:///Users/liyanran/Desktop/Andrew/function-space-variational-inference-yanran
  Preparing metadata (setup.py) ... done
Installing collected packages: fsvi
  Running setup.py develop for fsvi
Successfully installed fsvi-0.1
```
### toy objective function --BO
```
cd ../..
which python #确保运行的是python3
python run_base.py --data_training bayesian_optimization --model mlp_fsvi --architecture fc_100_100 --activation tanh --learning_rate 1e-3 --optimizer adam --batch_size 0 --prior_mean 0 --prior_cov 4 --prior_cov_offset 1 --prior_type bnn_induced --n_inducing_inputs 100 --n_marginals 1 --inducing_inputs_bound -10 10 --inducing_input_type uniform_rand_tdvi --kl_scale none --n_samples 5 --n_samples_eval 100 --logging_frequency 1000 --seed 0 --debug --save --save_path tmp --feature_map_jacobian --feature_map_type learned_grad --grad_flow_jacobian --full_cov --prior_mean_init --noise_std 0.01 --epochs 10000
```

### Ginsberg 指令
```
conda activate diva
fish
squeue -u $USER
squeue --user=yl5465
sbatch train.sh
bash srun_gpu.sh # request interactive gpu for debug
python sr.py -p train -c config/sr_wave_64_512CBIS.json -enable_wandb -log_wandb_ckpt -log_eval    #在interactive下call python
```

# Debug必备

## 申请cpu/gpu 
```
bash srun_cpu.sh
bash srun_gpu.sh
```

## Ginsburg
打断点 会在断点的地方停下
```
breakpoint()
```
停了之后可以查看各个变量名 e.g.
```
opt
```
如果看不清
```
(Pdb) opt['dataset']
(Pdb) from pprint import pprint
(Pdb) pprint(opt)
```
examples:
```
logger
```
输出： <Logger base (INFO)>
```
logger.name
```
'base'
```
type(opt)
```
<class 'core.logger.NoneDict'>

```
dir(opt)
```
['__class__', '__class_getitem__', '__contains__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__ior__', '__iter__', '__le__', '__len__', '__lt__', '__missing__', '__module__', '__ne__', '__new__', '__or__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__ror__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']
```
opt.keys()
```
dict_keys(['name', 'phase', 'gpu_ids', 'path', 'datasets', 'model', 'train', 'wandb', 'distributed', 'log_wandb_ckpt', 'log_eval', 'enable_wandb'])

回到terminal主界面： crtl+D















