# -*- coding: utf-8 -*-
from django.db import models
import uuid


class User(models.Model):
    guid = models.CharField(max_length=100, blank=False,
                            unique=True, default=uuid.uuid4)
    name = models.CharField(max_length=100, blank=True,
                            unique=False, default="Names not yet implemented")

def get_upload_dir(instance, filename):
    return '{0}/{1}'.format(('Documents/' + str(instance.user.pk)), filename)


class Document(models.Model):
    user = models.ForeignKey(User, default=999, on_delete=models.CASCADE)
    docfile = models.FileField(upload_to=get_upload_dir)
