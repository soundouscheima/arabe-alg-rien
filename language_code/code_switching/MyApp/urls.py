from django.urls import path
from . import views

urlpatterns = [
    path('', views.wanlp, name='wanlp'), 
    path('analyze/', views.analyze_text, name='analyze_text'), # your main page for /wanlp/
    path('ai-expert/', views.ai_expert, name='ai_expert'),
]