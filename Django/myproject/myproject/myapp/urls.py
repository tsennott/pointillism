# -*- coding: utf-8 -*-
from django.conf.urls import url
from myproject.myapp.views import list
from myproject.myapp.views import show_guid

urlpatterns = [
    url(r'^list/$', list, name='list'),
    url(r'^(?P<guid_id>[0-9]+)$', show_guid, name='show_guid')
]
