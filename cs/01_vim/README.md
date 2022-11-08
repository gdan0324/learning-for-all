# Vim

Linux种的vim

![image-20221028203146006](C:/Users/96212/AppData/Roaming/Typora/typora-user-images/image-20221028203146006.png)

- `:q` quit (close window)

- `:w` save (“write”)

- `:wq` save and quit

- `:e {name of file}` open file for editing

- `:ls` show open buffers

- `:help` {topic}

  open help

  - `:help :w` opens help for the `:w` command
  - `:help w` opens help for the `w` movement



`<ESC>`进入vim模式

`<i>`插入模式

`<R>`更改当前字符

`<v>`扫描



## 移动

- 基本动作：（`hjkl`左、下、上、右）

- 单词：（`w`下一个单词），`b`（单词开头），`e`（单词结尾）

- 行：（`0`行首），`^`（第一个非空白字符），`$`（行尾） 【好像不太行】

- 屏幕：（屏幕`H`顶部），`M`（屏幕中间），`L`（屏幕底部）

- 滚动：（`Ctrl-u`上）、`Ctrl-d`（下）

- 文件：（`gg`文件开头），`G`（文件结尾）

- 行号：`:{number}<CR>`或`{number}G`(line {number})

- 杂项：（`%`对应项目）

- 查找：f{character}、t{character}、F{character}、T{character} 【好像不太行】

  ​	在当前行查找/向前/向后{字符}

  ​	`,`/`;`用于导航匹配

- Search: `/{regex}`, `n`/`N`用于导航匹配【好像不太行】



## 选择

视觉模式：

- 视觉的：`v`
- 视觉线：`V`
- 视觉块：`Ctrl-v`

可以使用移动键进行选择。



## 编辑

- `o`/`O`在下方/上方插入行

- d{motion}

  删除{动作}

  - eg`dw`是删除词，`d$`是删除到行尾，`d0`是删除到行首

- c{motion}

  改变{动作}

  - 例如`cw`是换词
  - 像`d{motion}`其次`i`

- `x`删除字符（等于 do `dl`）

- `s`替代字符（等于`cl`）

- `u`撤消，`<C-r>`重做

- `y`复制/“yank”（一些其他命令，例如`d`复制）

- `p`粘贴



## 计数

- `3w`向前移动 3 个字
- `5j`下移 5 行
- `7dw`删除 7 个字



## 修饰符

- `ci(`更改当前括号内的内容
- `ci[`更改当前一对方括号内的内容
- `da`删除单引号字符串，包括周围的单引号



































