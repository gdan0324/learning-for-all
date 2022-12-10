public interface UF {
    // 查看 元素 p 和 q 是否连接。
    boolean isConnected(int p, int q);

    // 合并元素 p 和 q
    void unionElements(int p, int q);

    //获取当前并查集中有多少个元素
    int size();
}
