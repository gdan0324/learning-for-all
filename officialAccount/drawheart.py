import turtle



turtle.color("red","pink")

turtle.left(140)

turtle.forward(111.65)



turtle.begin_fill()



for i in range(200):

    turtle.right(1)

    turtle.forward(1)



turtle.left(120)





for i in range(200):

    turtle.right(1)

    turtle.forward(1)



turtle.forward(111.65)

turtle.end_fill()

turtle.write("Hello ,", font=("宋体", 38))



turtle.done()