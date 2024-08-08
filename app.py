from flask import Flask,render_template,request
import knn_model as model


app=Flask(__name__)

@app.route("/",methods=["POST","GET"])
def home():
    return render_template("home.html")




@app.route("/predict",methods=["POST","GET"])
def predict():
    res=None
    if request.method=="POST":
        s_len=request.form['sl']
        s_wid=request.form['sw']
        p_len=request.form['pl']
        p_wid=request.form['pw']

        result=model.predictor(s_len,s_wid,p_len,p_wid)

        return render_template("res.html",result=result)
    return render_template("home.html")
    
if __name__=="__main__":
    app.run(debug=True)    


