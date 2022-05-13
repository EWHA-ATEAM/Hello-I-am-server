from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('api/sentence-label/', views.label_user_chat),
    path('api/image-test/', views.get_image),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
