from django.urls import path
from .views import FlowerDetectionView

urlpatterns = [
    path('detect/', FlowerDetectionView.as_view(), name='flower_detect'),
]
