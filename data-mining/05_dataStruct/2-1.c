#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXSIZE 8
int board[MAXSIZE][MAXSIZE];
int t = 1;

void ChessBoard(int tr, int tc, int dr, int dc, int size)
{
    int s, t1; // t1表示本次覆盖所用L型骨牌的编号
    if (size == 1)
        return;                        //棋盘只有一个方格且是特殊方格
    t1 = ++t;                          // t为全局变量，表示 L型骨牌编号
    s = size / 2;                      // 划分棋盘
    if (dr < tr + s && dc < tc + s)    //特殊方格在左上角子棋盘中
        ChessBoard(tr, tc, dr, dc, s); //递归处理子棋盘
    else
    { //用 t 号L型骨牌覆盖其右下角，再递归处理子棋盘
        board[tr + s - 1][tc + s - 1] = t1;
        ChessBoard(tr, tc, tr + s - 1, tc + s - 1, s);
    }
    if (dr < tr + s && dc >= tc + s) //特殊方格在右上角子棋盘中
        ChessBoard(tr, tc + s, dr, dc, s);
    else
    { //用 t 号L型骨牌覆盖其左下角，再递归处理子棋盘
        board[tr + s - 1][tc + s] = t1;
        ChessBoard(tr, tc + s, tr + s - 1, tc + s, s);
    }

    if (dr >= tr + s && dc < tc + s) //特殊方格在左下角子棋盘中
        ChessBoard(tr + s, tc, dr, dc, s);
    else
    { //用 t 号L型骨牌覆盖其右上角，再递归处理子棋盘
        board[tr + s][tc + s - 1] = t1;
        ChessBoard(tr + s, tc, tr + s, tc + s - 1, s);
    }
    if (dr >= tr + s && dc >= tc + s) //特殊方格在右下角子棋盘中
        ChessBoard(tr + s, tc + s, dr, dc, s);
    else
    { //用 t 号L型骨牌覆盖其左上角，再递归处理子棋盘
        board[tr + s][tc + s] = t1;
        ChessBoard(tr + s, tc + s, tr + s, tc + s, s);
    }
}
int main()
{
    int i, j;
    board[0][1] = 0;
    int a;
    int b;
    printf( "Enter row and col :");
    scanf("%d %d", &a, &b);
    ChessBoard(0, 0, a, b, MAXSIZE);
    for (i = 0; i < MAXSIZE; i++)
    {
        for (j = 0; j < MAXSIZE; j++)
            printf("%5d", board[i][j]);
        printf("\n");
    }
    return 0;
}