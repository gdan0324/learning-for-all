public class UnionFind1 implements UF{
    // id 数组即刚才我们演示过程中的数组
    private int[] id;

    public UnionFind1(int size){

        //在构造函数中为 id 数组初始化一个长度 size
        id = new int[size];

        // 在为归并任何两个顶点之前，每个顶点单独在一个集合中
        // 初始化 id 数组， 即元素 i 在第 i 号集合
        for(int i = 0 ; i < id.length ; ++i)
            id[i] = i;
    }

    // 用于查找顶点 p 也就是元素 p 所在集合的编号
    private int find(int p){
        // 如果顶点 p 超出了我们一开始所给的 size 范围
        // 抛出异常
        if(p < 0 || p >= id.length)
            throw new IllegalArgumentException("p is out of bound");
        //返回顶点 p 的集合编号
        return id[p];
    }

    @Override
    public boolean isConnected(int p, int q) {
        // 查看顶点 p 和 顶点 q 是否连接
        // 连接则返回 true 不连接 则返回 false
        return id[p] == id[q];
    }

    @Override
    public void unionElements(int p, int q) {
        // 如果 顶点 p 和顶点 q 已经连接
        // 则什么都不用做，直接 结束函数就好
        if(id[p] == id[q])
            return;
        // 查找 顶点 p 和 顶点 q 所在集合编号
        int pID = find(p);
        int qID = find(q);
        // 将与 顶点 q 相连的所有顶点以及 q 顶点
        // 放入 顶点p 所在的集合中
        for(int i = 0 ; i < id.length ; ++i)
            if(id[i] == qID)
                id[i] = pID;

    }

    @Override
    public int size() {
        return id.length;
    }

    public void showArray(){

        for(int i = 0 ; i < id.length ; ++i)
            System.out.print(id[i] + "  ");
    }

}
