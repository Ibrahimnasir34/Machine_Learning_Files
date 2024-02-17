def update_theta(x,y,y_hat,b_0, theta_o, learning_rate):

    db=(np.sum(y_hat-y)*2)/Len(y)

    dw=(np.dot((y_hat-y),x)*2)/len(y)

    b_1=b_0-learning_rate*db

    theta_1=theta_o-learning_ratexdw

    return b_1, theta_1

print("After initialization -Bias: ",b, "theta: ", theta),

Y_hat-predict_Y(b, theta, X)

b, theta-update_theta(X,Y,Y_hat, b, theta,0.01)

print("After first update -Bias: "b, "theta: ", theta)

get_cost(Y,Y_hat)