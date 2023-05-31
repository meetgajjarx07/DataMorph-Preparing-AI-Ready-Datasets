import secrets
import asyncio
import os.path
import numpy as np
import pandas as pd

from shutil import copyfile
from flask import *
from skylearn.preprocessing import generic_preprocessing as gp
from skylearn.modules import logistic as lg
from skylearn.modules import naive_bayes as nb
from skylearn.modules import linear_svc as lsvc
from skylearn.modules import knn
from skylearn.modules import decision_tree as dtree
from skylearn.modules import random_forest as rfc
from skylearn.visualization import visualize as vis
from skylearn.nocache import nocache
from skylearn import app
global posted
import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response
from datetime import datetime
from functools import wraps, update_wrapper

save_path = "skylearn/uploads/"
exts = ["csv", "json", "yaml"]
ext_img = ["jpeg","jpg","png"]
posted = 0


@app.route("/", methods=["GET", "POST"])
@app.route("/preprocess", methods=["GET", "POST"])

def preprocess():

	if request.method == "POST":
		
		try:
			hid_tag = request.form.get("hid")
			# print(hid_tag)
			if request.files["file_up"].filename == '':
				return redirect('/')
	
			if hid_tag == 'hid_page':
				data = request.files["file_up"]
				ext = data.filename.split(".")[1]
				print(data.filename)
				

				if ext in exts:
					session["ext"] = ext
					session["fname"] = data.filename
					session['page'] = 'pg_csv'
					data.save("skylearn/uploads/" + data.filename)
					df = gp.read_dataset("skylearn/uploads/" + data.filename)
					df.to_csv("skylearn/clean/clean.csv")
					session["haha"] = True
					return redirect(url_for('preprocess'))
	
	
				elif ext in ext_img:
					session["ext"] = ext
					session["fname"] = 'temp_img'
					data.save("skylearn/static/img/temp_img." + ext)
					return redirect(url_for('ImagePro'))
	
				else:
					flash(f"Upload Unsuccessful. Please try again", "danger")
		except:

			if request.form["Submit"] == "Delete":
				try:
					df = gp.read_dataset("skylearn/clean/clean.csv")
					df = gp.delete_column(df, request.form.getlist("check_cols"))
					df.to_csv("skylearn/clean/clean.csv", mode="w", index=False)
					flash(f"Column(s) deleted Successfully", "success")
				except:
					flash(f"Column(s) deleted Successfully", "success")

			elif request.form["Submit"] == "Clean":
				try:
					df = gp.read_dataset("skylearn/clean/clean.csv")
					print(request.form["how"])
					if request.form["how"] != "any":
						df = gp.treat_missing_numeric(
							df, request.form.getlist("check_cols"), how=request.form["how"]
						)
					elif request.form["howNos"] != None:
						df = gp.treat_missing_numeric(
							df,
							request.form.getlist("check_cols"),
							how=float(request.form["howNos"]),
						)

					df.to_csv("skylearn/clean/clean.csv", mode="w", index=False)
					flash(f"Column(s) cleant Successfully", "success")
				except:
					flash(f"Column(s) cleant Successfully", "success")

			elif request.form["Submit"] == "Visualize":
				global posted
				df = gp.read_dataset("skylearn/clean/clean.csv")

				x_col = request.form["x_col"]

				if vis.hist_plot(df, x_col):
					posted = 1

	if session.get("haha") != None:
		df = gp.read_dataset("skylearn/clean/clean.csv")
		description = gp.get_description(df)
		columns = gp.get_columns(df)
		print(columns)
		dim1, dim2 = gp.get_dim(df)
		head = gp.get_head(df)

		return render_template(
			"preprocess.html",
			active="preprocess",
			title="Preprocess",
			filename=session["fname"],
			posted=posted,
			no_of_rows=len(df),
			no_of_cols=len(columns),
			dim=str(dim1) + " x " + str(dim2),
			description=description.to_html(
				classes=[
					"table-bordered",
					"table-striped",
					"table-hover",
					"thead-light",
				]
			),
			columns=columns,
			head=head.to_html(
				classes=[
					"table",
					"table-bordered",
					"table-striped",
					"table-hover",
					"thead-light",
				]
			),
		)
	else:
		return render_template("index.html", active="preprocess", title="Preprocess")


@app.route("/classify", methods=["GET", "POST"])
def classify():
	acc = 0
	if request.method == "POST":
		target = request.form["target"]
		gp.arrange_columns(target)
		classifier = int(request.form["classifier"])
		hidden_val = int(request.form["hidden"])
		scale_val = int(request.form["scale_hidden"])
		encode_val = int(request.form["encode_hidden"])
		columns = vis.get_columns()

		if hidden_val == 0:
			data = request.files["choiceVal"]
			ext = data.filename.split(".")[1]
			if ext in exts:
				data.save("skylearn/uploads/test." + ext)
			else:
				return "File type not accepted!"
			choiceVal = 0
		else:
			choiceVal = int(request.form["choiceVal"])

		if classifier == 0:
			ret_vals = lg.logisticReg(choiceVal, hidden_val, scale_val, encode_val)
			if hidden_val == 0 or hidden_val == 1:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=[
						ret_vals[1].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					conf_matrix=[
						ret_vals[2].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
				)
			elif hidden_val == 2:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=ret_vals[1],
					conf_matrix=ret_vals[2],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)

		elif classifier == 1:
			ret_vals = nb.naiveBayes(choiceVal, hidden_val, scale_val, encode_val)
			if hidden_val == 0 or hidden_val == 1:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=[
						ret_vals[1].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					conf_matrix=[
						ret_vals[2].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)
			elif hidden_val == 2:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=ret_vals[1],
					conf_matrix=ret_vals[2],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)

		elif classifier == 2:
			ret_vals = lsvc.lin_svc(choiceVal, hidden_val, scale_val, encode_val)
			if hidden_val == 0 or hidden_val == 1:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=[
						ret_vals[1].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					conf_matrix=[
						ret_vals[2].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)
			elif hidden_val == 2:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=ret_vals[1],
					conf_matrix=ret_vals[2],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)

		elif classifier == 3:

			scale_val = 1
			ret_vals = knn.KNearestNeighbours(
				choiceVal, hidden_val, scale_val, encode_val
			)
			if hidden_val == 0 or hidden_val == 1:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=[
						ret_vals[1].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					conf_matrix=[
						ret_vals[2].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)
			elif hidden_val == 2:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=ret_vals[1],
					conf_matrix=ret_vals[2],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)

		elif classifier == 4:
			ret_vals = dtree.DecisionTree(choiceVal, hidden_val, scale_val, encode_val)
			if hidden_val == 0 or hidden_val == 1:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=[
						ret_vals[1].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					conf_matrix=[
						ret_vals[2].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)
			elif hidden_val == 2:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=ret_vals[1],
					conf_matrix=ret_vals[2],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)
		elif classifier == 5:
			ret_vals = rfc.RandomForest(choiceVal, hidden_val, scale_val, encode_val)
			if hidden_val == 0 or hidden_val == 1:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=[
						ret_vals[1].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					conf_matrix=[
						ret_vals[2].to_html(
							classes=[
								"table",
								"table-bordered",
								"table-striped",
								"table-hover",
								"thead-light",
							]
						)
					],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)
			elif hidden_val == 2:
				return render_template(
					"classifier_page.html",
					acc=ret_vals[0],
					report=ret_vals[1],
					conf_matrix=ret_vals[2],
					choice=hidden_val,
					classifier_used=classifier,
					active="classify",
					title="Classify",
					cols=columns,
				)
	elif request.method == "GET":
		columns = vis.get_columns()
		return render_template(
			"classifier_page.html", active="classify", title="Classify", cols=columns
		)



@app.route("/clear", methods=["GET"])
def clear():
	session.clear()
	return redirect("/")


@app.route("/visualize", methods=["GET", "POST"])
def visualize():
	
	if request.method == "POST":
		x_col = request.form["x_col"]
		y_col = request.form["y_col"]
	
		df = vis.xy_plot(x_col, y_col)
		heights = np.array(df[x_col]).tolist()
		weights = np.array(df[y_col]).tolist()
	
		newlist = []
		for h, w in zip(heights, weights):
			newlist.append({"x": h, "y": w})
		ugly_blob = str(newlist).replace("'", "")
	
		columns = vis.get_columns()
		print(x_col)
		return render_template(
			"visualize.html",
			cols=columns,
			src="static/img/pairplot1.png",
			xy_src="static/img/fig.png",
			posted=1,
			data=ugly_blob,
			active="visualize",
			x_col_name=str(x_col),
			y_col_name=str(y_col),
			title="Visualize",
		)
	
	else:
		vis.pair_plot()
		columns = vis.get_columns()
		return render_template(
			"visualize.html",
			cols=columns,
			src="static/img/pairplot1.png",
			posted=0,
			active="visualize",
			title="Visualize",
		)
	



@app.route("/col.csv")
def col():
	return send_file("visualization/col.csv", mimetype="text/csv", as_attachment=True)


@app.route("/pairplot1.png")

def pairplot1():
	return send_file(
		"static/img/pairplot1.png", mimetype="image/png", as_attachment=True
	)


@app.route("/tree.png")

def tree():
	return send_file("static/img/tree.png", mimetype="image/png", as_attachment=True)







@app.route("/imgpro", methods=["GET", "POST"])
def ImagePro():
	if request.method == "GET":
		return render_template("ImgProcess/uploaded.html",file_path="img/temp_img.jpeg")
		# return render_template("imagedi.html")

@app.route("/normal", methods=["POST"])
@nocache
def normal():
    return render_template("ImgProcess/uploaded.html", file_path="img/temp_img.jpeg")


@app.route("/brightness")
def brightness():
    return render_template("ImgProcess/brightness.html", file_path="img/temp_img.jpeg")


@app.route("/darkening")
def darkening():
    return render_template("ImgProcess/darkening.html", file_path="img/temp_img.jpeg")



@app.route("/grayscale", methods=["POST"])
def grayscale():
    image_processing.grayscale()
    return render_template("ImgProcess/uploaded.html", file_path="img/temp_img_grayscale.jpeg")


@app.route("/inverse", methods=["POST"])
def inverse():
    image_processing.invers()
    return render_template("ImgProcess/uploaded.html", file_path="img/temp_img_inverse.jpeg")


@app.route("/fliphorizontal", methods=["POST"])
def fliphorizontal():
    image_processing.fliphorizontal()
    return render_template("ImgProcess/uploaded.html", file_path="img/temp_img_fliphorizontal.jpeg")

@app.route("/flipvertical", methods=["POST"])
def flipvertical():
    image_processing.flipvertical()
    return render_template("ImgProcess/uploaded.html", file_path="img/temp_img_flipvertical.jpeg")


@app.route("/brightnesswithincrease", methods=["POST"])
@nocache
def brightnesswithincrease():
    val = request.form['val_increase']
    image_processing.brightnesswithincrease(val)
    return render_template("ImgProcess/brightness.html", file_path="img/temp_img_brightnesswithincrease.jpeg")


@app.route("/brightnesswithmultiply", methods=["POST"])
@nocache
def brightnesswithmultiply():
    val = request.form['val_multiply']
    image_processing.brightnesswithmultiply(val)
    return render_template("ImgProcess/brightness.html", file_path="img/temp_img_brightnesswithmultiply.jpeg")

@app.route("/darkeningwithdecrease", methods=["POST"])
@nocache
def darkeningwithdecrease():
    val = request.form['val_increase']
    image_processing.darkeningwithdecrease(val)
    return render_template("ImgProcess/darkening.html", file_path="img/temp_img_darkeningwithdecrease.jpeg")


@app.route("/darkeningwithdivide", methods=["POST"])
@nocache
def darkeningwithdivide():
    val = request.form['val_multiply']
    image_processing.darkeningwithdivide(val)
    return render_template("ImgProcess/darkening.html", file_path="img/temp_img_darkeningwithdivide.jpeg")


@app.route("/convolution")
@nocache
def convolution():
    return render_template("ImgProcess/convolution.html", file_path="img/temp_img.jpeg")


@app.route("/histogram")
@nocache
def histogram():
    image_processing.histogram()
    return render_template("ImgProcess/histogram.html")


@app.route("/blurring")
def blurring():
    blur_pix = 1 / 9
    image_processing.convolute(blur_pix, blur_pix, blur_pix,
                               blur_pix, blur_pix, blur_pix, blur_pix, blur_pix, blur_pix, "")
    return render_template("ImgProcess/uploaded.html", file_path="img/temp_img_convolution.jpeg")


@app.route("/sharpening")
def sharpening():
    image_processing.convolute(0, -1, 0, -1, 5, -1, 0, -1, 0, "")
    return render_template("ImgProcess/uploaded.html", file_path="img/temp_img_convolution.jpeg")

@app.route("/edge_detection")
def edge_detection():
    image_processing.convolute(-1, -1, -1, -1, 8, -1, -1, -1, -1, "edge")
    return render_template("ImgProcess/uploaded.html", file_path="img/temp_img_convolution.jpeg")

@app.route("/convoluting", methods=["POST"])
def convoluting():
    m11 = request.form['mat11']
    m12 = request.form['mat12']
    m13 = request.form['mat13']
    m21 = request.form['mat21']
    m22 = request.form['mat22']
    m23 = request.form['mat23']
    m31 = request.form['mat31']
    m32 = request.form['mat32']
    m33 = request.form['mat33']

    try:
        image_processing.convolute(
            m11, m12, m13, m21, m22, m23, m31, m32, m33, "ordinary")
    except:
        return render_template("ImgProcess/convolution.html", file_path="img/temp_img.jpeg", alert="Matrix must filled all by integers")
    return render_template("ImgProcess/convolution.html", file_path="img/temp_img_convolution.jpeg")





