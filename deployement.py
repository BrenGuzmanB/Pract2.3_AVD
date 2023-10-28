"""
Created on Fri Oct 27 00:47:15 2023

@author: Brenda García
"""

# FASE VI. DESPLIEGUE (DEPLOYMENT)
#pip install gradio
import gradio as gr
import numpy as np
import joblib

# Define diccionarios para mapear 
star_mapping = {
    "5 estrellas": 5,
    "4 estrellas": 4,
    "3 estrellas": 3,
    "2 estrellas": 2,
    "1 estrella": 1
}

customer_mapping = {
    "Cliente fiel": 0,
    "Cliente nuevo/ocasional": 1
    
}

class_mapping = {
    "Eco": 0,
    "Eco Plus": 2,
    "Negocios": 1
    
}

def predict_satisfaction(inflight_entertainment, ease_of_online_booking, online_support, on_board_service, seat_comfort, online_boarding, leg_room_service, customer_type, class_, baggage_handling, checkin_service, cleanliness, inflight_wifi_service):
    # Mapea las calificaciones en estrellas a números
    inflight_entertainment = star_mapping.get(inflight_entertainment, inflight_entertainment)
    
    ease_of_online_booking = star_mapping.get(ease_of_online_booking, ease_of_online_booking)
    
    online_support = star_mapping.get(online_support, online_support)
    
    on_board_service = star_mapping.get(on_board_service, on_board_service)
    
    seat_comfort = star_mapping.get(seat_comfort, seat_comfort)
    
    online_boarding = star_mapping.get(online_boarding, online_boarding)
    
    leg_room_service = star_mapping.get(leg_room_service, leg_room_service)
    
    customer_type = customer_mapping.get(customer_type, customer_type)
    
    class_ = class_mapping.get(class_, class_)
    
    baggage_handling = star_mapping.get(baggage_handling, baggage_handling)
    
    checkin_service = star_mapping.get(checkin_service, checkin_service)
    
    cleanliness = star_mapping.get(cleanliness, cleanliness)
    
    inflight_wifi_service = star_mapping.get(inflight_wifi_service, inflight_wifi_service)
    
    # Importar el modelo de regresión
    model = joblib.load("modelo.pkl")

    # Convertir las entradas a un vector
    inputs = [inflight_entertainment, ease_of_online_booking, online_support, on_board_service, seat_comfort, online_boarding, leg_room_service, customer_type, class_, baggage_handling, checkin_service, cleanliness, inflight_wifi_service]
    inputs = np.array(inputs)

    # Predecir la satisfacción
    satisfaction = model.predict(inputs.reshape(1, -1))[0]
    if satisfaction == 1:
        r = "Cliente satisfecho"
    else:
        r = "Cliente insatisfecho"

    return r


# Crear la interfaz
demo = gr.Interface(
    predict_satisfaction,
    [
        #inflight_entertainment
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #ease_of_online_booking
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #online_support
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #on_board_service
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #seat_comfort
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #online_boarding
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #leg_room_service
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #customer_type
        gr.Radio(["Cliente fiel", "Cliente nuevo/ocasional"]),
        #class_
        gr.Radio(["Eco", "Eco Plus", "Negocios"]),
        #baggage_handling
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #checkin_service
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #cleanliness
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
        #inflight_wifi_service
        gr.Radio(["5 estrellas", "4 estrellas", "3 estrellas", "2 estrellas", "1 estrella"]),
    ],
    "text",
    live=True,
    # Especificar la salida de la interfaz
    title="Aspectos que influyen en la satisfacción de los clientes",

)

# Lanzar la interfaz
demo.launch()
