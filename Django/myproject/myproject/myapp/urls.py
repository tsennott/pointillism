# -*- coding: utf-8 -*-
from django.conf.urls import url
from myproject.myapp.views import list
from myproject.myapp.views import new_guid

urlpatterns = [
    url(r'^$', new_guid, name='new_guid'),
    url(r'^(?P<guid_id>[0-9]+)$', list, name='list')
]
