# -*- coding: utf-8 -*-
from django.db import models
import uuid


class User(models.Model):
    guid = models.CharField(max_length=100, blank=True,
                            unique=True, default=uuid.uuid4)


def get_upload_dir(instance, filename):
    return '{0}/{1}'.format(('Documents/' + str(instance.user.guid)), filename)


class Document(models.Model):
    user = models.ForeignKey(User, default=uuid.uuid4, on_delete=models.CASCADE)
    docfile = models.FileField(upload_to=get_upload_dir)
