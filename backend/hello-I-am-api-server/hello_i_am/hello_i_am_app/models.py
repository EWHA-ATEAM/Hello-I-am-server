from django.db import models

import os
from django.conf import settings

from django.db.models.signals import post_delete
from django.dispatch import receiver

class Screenimage(models.Model):
    image = models.ImageField(upload_to='images/')
    # models에 ImageField를 통해 이미지를 받을 수 있도록
    # 사용자가 업로드한 이미지가 들어가게 됩니다.
    # 이때, MEDIA_ROOT를 지정해줌으로서, 해당 경로에 이미지 사진을 저장
    # upload_to 옵션을 넣은 이유는 BASE_DIR/media/images/ 아래에 저장하게 하기 위해


@receiver(post_delete, sender=Screenimage)
def screenimage_delete_action(sender, instance, **kwargs):
    instance.image.delete(False)
