# Git

## 1. 版本控制

版本控制是一种记录文作内容变化以便将来查阅特定版本修订情况的系统。
版本控制其实最重要的是可以记录文件修改历史记录，从而让用户能够查看历史版本，方便版本切换。

## 2. 集中式和分布式

先说集中式版本控制系统，版本库是集中存放在中央服务器的，而干活的时候，用的都是自己的电脑，所以要先从中央服务器取得最新的版本，然后开始干活，干完活了，再把自己的活推送给中央服务器。中央服务器就好比是一个图书馆，你要改一本书，必须先从图书馆借出来，然后回到家自己改，改完了，再放回图书馆。

![central-repo](https://www.liaoxuefeng.com/files/attachments/918921540355872/l)

集中式版本控制系统最大的毛病就是必须联网才能工作，如果在局域网内还好，带宽够大，速度够快，可如果在互联网上，遇到网速慢的话，可能提交一个10M的文件就需要5分钟，这还不得把人给憋死啊。

那分布式版本控制系统与集中式版本控制系统有何不同呢？首先，<u>分布式版本控制系统根本没有“中央服务器”</u>，<u>每个人的电脑上都是一个完整的版本库</u>，这样，你工作的时候，就不需要联网了，因为版本库就在你自己的电脑上。既然每个人电脑上都有一个完整的版本库，那多个人如何协作呢？比方说你在自己电脑上改了文件A，你的同事也在他的电脑上改了文件A，这时，你们俩之间只需把各自的修改推送给对方，就可以互相看到对方的修改了。

和集中式版本控制系统相比，分布式版本控制系统的安全性要高很多，因为每个人电脑里都有完整的版本库，某一个人的电脑坏掉了不要紧，随便从其他人那里复制一个就可以了。而集中式版本控制系统的中央服务器要是出了问题，所有人都没法干活了。

在实际使用分布式版本控制系统的时候，其实很少在两人之间的电脑上推送版本库的修改，因为可能你们俩不在一个局域网内，两台电脑互相访问不了，也可能今天你的同事病了，他的电脑压根没有开机。因此，分布式版本控制系统通常也有一台充当“中央服务器”的电脑，但这个服务器的作用仅仅是用来方便“交换”大家的修改，没有它大家也一样干活，只是交换修改不方便而已。

![distributed-repo](https://www.liaoxuefeng.com/files/attachments/918921562236160/l)

当然，Git的优势不单是不必联网这么简单，后面我们还会看到Git极其强大的分支管理，把SVN等远远抛在了后面。



## 3. 工作机制

![image-20221028124658876](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221028124658876.png)

​	远程库：

​		GtiHub——国外

​		Gitee——国内

​		GitLab——局域网



## 4. 常用命令

| 命令名称                             | 作用           |
| ------------------------------------ | -------------- |
| git config --global user.name 用户名 | 设置用户签名   |
| git config --global user.email 邮箱  | 设置用户签名   |
| git init                             | 初始化本地库   |
| git status                           | 查看本地库状态 |
| git add 文件名                       | 添加到暂存区   |
| git commit -m "日志信息" 文件名      | 提交到本地库   |
| git reflog                           | 查看历史记录   |
| git reset --hard 版本号              | 版本穿梭       |



### 4.1 git init



<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221028201609502.png" alt="image-20221028201609502" style="zoom:80%;" />

查看隐藏目录

![image-20221028201855471](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221028201855471.png)



### 4.2 git status



![image-20221028202216535](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221028202216535.png)

On branch master ：当前master分支

No commits yet ：还没有任何提交

nothing added to commit but untracked files present (use "git add" to track) ：证明有文件没有添加到暂存区

#### cat  hello.txt 

​	查看文件内容

#### tail -n 1 hello.txt

​	查看hello.txt末尾

```Git
LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo
$ git status
fatal: not a git repository (or any of the parent directories): .git

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo
$ git init
Initialized empty Git repository in G:/git-demo/.git/

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ ll
total 0

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ ll -a
total 8
drwxr-xr-x 1 LD 197121 0 Oct 28 20:44 ./
drwxr-xr-x 1 LD 197121 0 Oct 28 20:44 ../
drwxr-xr-x 1 LD 197121 0 Oct 28 20:44 .git/

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ vim hello.txt

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ git status
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        hello.txt

nothing added to commit but untracked files present (use "git add" to track)

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ git add hello.txt
warning: LF will be replaced by CRLF in hello.txt.
The file will have its original line endings in your working directory

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ git status
On branch master

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   hello.txt

```



#### 移除和添加暂存区

git rm --cache hello.txt

git add 

![image-20221029094220647](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221029094220647.png)



#### 将暂存区文件提交到本地库

​	git commit -m "日志信息" 文件名

​	

```Git
LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ git commit -m "first" hello.txt
warning: LF will be replaced by CRLF in hello.txt.
The file will have its original line endings in your working directory
[master (root-commit) 6e220d1] first
 1 file changed, 11 insertions(+)
 create mode 100644 hello.txt

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ git status
On branch master
nothing to commit, working tree clean

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ git reflog
6e220d1 (HEAD -> master) HEAD@{0}: commit (initial): first

LD@LAPTOP-9LJ1TK4J MINGW64 /g/git-demo (master)
$ git log
commit 6e220d157f6b05460846626764c0e0f42b2806d5 (HEAD -> master)
Author: gdan <962127853@qq.com>
Date:   Sat Oct 29 10:18:55 2022 +0800

    first

```

6e220d1 代表前七位精简版本号，全部显示是6e220d157f6b05460846626764c0e0f42b2806d5



![image-20221029102746739](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221029102746739.png)

1. git reflog
2. git status
3. vim hello.txt
4. git add hello.txt
5. git commit -m "third submit" hello.txt
6. git reflog
7. git log



#### 版本穿梭

![image-20221029104121697](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221029104121697.png)

hot-fix——热修



#### 分支操作

| 命令名称            | 作用                         |
| ------------------- | ---------------------------- |
| git branch 分支名   | 创建分支                     |
| git branch -v       | 查看分支                     |
| git checkout 分支名 | 切换分支                     |
| git merge 分支名    | 把指定的分支合并到当前分支上 |

1. git branch -v
2. git branch hot-fix
3. git branch -v
4. git checkout hot-fix
5. vim hello.txt  修改
6. git add hello.txt
7. git commit -m 'hot fix submit' hello.txt
8. git status
9. git reflog

<img src="C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221029153745533.png" alt="image-20221029153745533" style="zoom:80%;" />

#### 合并操作

master分支没有修改，

1. git checkout hot-fix
2. vim hello.txt
3. git add
4. git commit -m "hot fix submit" 
5. git checkout master
6. git  merge

master分支和hot-fix分支同一行代码有修改->冲突

1. git checkout master
2. vim hello.txt （修改最后一行信息）
3. git add
4. git commit -m "master" hello.txt
5. git checkout hot-fix
6. vim hello.txt（修改最后一行信息）
7. git add hello.txt
8. git commit -m "hot" hello.txt
9. git checkout master
10. git merge hot-fix

![image-20221030091619120](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221030091619120.png)

MERGING证明没有合并成功

![image-20221030091720967](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221030091720967.png)



![image-20221030091946069](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221030091946069.png)

将横线划出来的删除，删除后再提交，要注意的是提交的时候后面不要带文件名。

![image-20221030093211657](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221030093211657.png)

此时可以查看hot-fix分支，该分支上是没有master分支上修改的代码。

简单的说就是，git玩的就是两个指针，一个是HEAD指针，第二个是master指针。





