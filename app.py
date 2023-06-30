from flask import Flask, request, render_template, jsonify, Response
from flask_cors import CORS, cross_origin
from CarPrice.exception import CarPriceException
from CarPrice.logger import logging
from CarPrice.constant.application import APP_HOST, APP_PORT
from CarPrice.pipeline.prediction_pipeline import CarData, CarPriceRegressor
from CarPrice.pipeline.training_pipeline import TrainingPipeline
from CarPrice.utils.main_utils import get_carlist

application = Flask(__name__)
app = application

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict_datapoint():

    car_list = get_carlist()

    if request.method == 'GET':
        return render_template('index.html')
    else:

        carprice_data = CarData(
            car_name = request.form.get("car_name"),
            vehicle_age = int(request.form.get("vehicle_age")),
            km_driven = int(request.form.get("km_driven")),
            seller_type = request.form.get("seller_type"),
            fuel_type = request.form.get("fuel_type"),
            transmission_type = request.form.get("transmission"),
            mileage = float(request.form.get("mileage")),
            engine = int(request.form.get("engine")),
            max_power = float(request.form.get("max_power")),
            seats = int(request.form.get("seats"))
        )


        carprice_df = carprice_data.get_car_price_input_data_frame()

        print(carprice_df)

        model_predictor = CarPriceRegressor()

        car_value = model_predictor.predict(dataframe=carprice_df)[0]

        results = round(car_value,2)

        return render_template('index.html', results=results, carprice_df=carprice_df)


@app.route("/train")
@cross_origin()
def trainRoute():
    try:
        pipeline = TrainingPipeline()

        pipeline.run_pipeline()
        
        return Respone("Training successful !!")

    except Exception as e:
        return Response(f"Error Occured ! {e}")

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)
