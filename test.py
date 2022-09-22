from __future__ import print_function
import json

from flask import request

from flask import Flask, render_template,redirect,url_for,request


from pipe import pipeline

app = Flask(__name__)


@app.route('/', methods=['POST','GET'])
def page1():
    return render_template("landing_page.html")
@app.route('/get_questions', methods=['POST','GET'])
def page2():
    
    if request.method=="POST":
        message=request.form["mes"]
        model=request.form["model"]
        alpha_1=request.form["alpha"]
    
        return redirect(url_for("home",name=message,model_name=model,alpha=alpha_1))
    else: 
        return render_template("get_questions.html")
@app.route("/compare", methods=['POST','GET'])
def page3():
    if request.method=="POST":
        message=request.form["mes"]
        model1=request.form["model1"]
        alpha_1=request.form["alpha1"]
        model2=request.form["model2"]
        alpha_2=request.form["alpha2"]
        #these values are extracted from the form.
        #these values need to be passed to the python code thats why redirecting to a different link called home2
        return redirect(url_for("home2",name=message,model_name1=model1,alpha1=alpha_1,model_name2=model2,alpha2=alpha_2))
    else: 
        return render_template("compare.html")

@app.route("/<name>/<model_name>/<alpha>",methods=['POST','GET'])
def home(name,model_name,alpha):
     
     nlp = pipeline(model_name,alpha)
     if(model_name=="base_bart_2" or model_name=="base_t5_7"):
        return render_template("get_questions_output.html",question=nlp(name),paragraph=name,model=model_name,alpha=alpha) 
     return render_template("get_questions_output.html",question=list(nlp(name)[0]),paragraph=name,explanation_1=nlp(name)[1],model=model_name,alpha=alpha) 
@app.route("/<name>/<model_name1>/<alpha1>/<model_name2>/<alpha2>",methods=['POST','GET'])
def home2(name,model_name1,alpha1,model_name2,alpha2):
   #here the values are being caught and python functions are called.
     nlp = pipeline(model_name1,alpha1)
     nlp2 = pipeline(model_name2,alpha2)
     questions1=list(nlp(name)[0])
     questions2=list(nlp2(name)[0])
     if(model_name1=="base_bart_2" or model_name1=="base_t5_7"):
        #here model name is checked because for baselines there is no representation being printed 
        #otherwise there is representation and questions both are printed which can be extracted using nlp(name)[0] and nlp(name)[1]
        questions1=nlp(name)
     if(model_name2=="base_bart_2" or model_name2=="base_t5_7"):
        questions2=nlp2(name)
     #here html page for showing the output is being rendered and passing the questions,sentence,and the explanation to the HTML page.
     return render_template("compare_model_output.html",question1=questions1,question2=questions2,paragraph=name,model1=model_name1,model2=model_name2,alpha1=alpha1,alpha2=alpha2,model_number_1=model_name1,model_number_2=model_name2,explanation_1=nlp(name)[1],explanation_2=nlp2(name)[1]) 

if __name__=="__main__":

    app.run()