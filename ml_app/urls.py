from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.training_results, name='training_results'),
    path('predict/', views.predict_rul, name='predict_rul'),
]
