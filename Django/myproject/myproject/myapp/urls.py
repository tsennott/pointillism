# -*- coding: utf-8 -*-
from django.conf.urls import url
from myproject.myapp.views import upload
from myproject.myapp.views import new_guid
from myproject.myapp.views import gallery
from myproject.myapp.views import info
from myproject.myapp.views import gif


urlpatterns = [
    url(r'^$', new_guid, name='new_guid'),
    # url(r'^(?P<guid_id>[0-9]+)$', upload, name='upload'),
    url(r'^upload/(?P<guid_id>[0-9]+)$', upload, name='upload'),
    url(r'^gallery/(?P<guid_id>[0-9]+)$', gallery, name='gallery'),
    url(r'^info/(?P<guid_id>[0-9]+)$', info, name='info'),
    url(r'^gif/(?P<guid_id>[0-9]+)$', gif, name='gif'),
]
