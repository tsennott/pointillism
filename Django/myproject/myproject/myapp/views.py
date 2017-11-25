# -*- coding: utf-8 -*-
from django.shortcuts import get_object_or_404, render
from django.template import RequestContext
from django.http import HttpResponseRedirect, HttpResponse
from django.core.urlresolvers import reverse

from django.views.decorators.csrf import csrf_exempt

from myproject.myapp.models import Document
from myproject.myapp.models import User
from myproject.myapp.forms import DocumentForm

from myproject.myapp.pointillism import pointillize, pointillizeStack
from PIL import Image
import os

import io
from django.core.files.uploadedfile import InMemoryUploadedFile


def new_guid(request):
    user = User()
    user.name = 'New User'
    user.save()
    # request.session['guid_id'] = user.pk
    return HttpResponseRedirect(reverse('gallery',
                                        kwargs={'guid_id': user.pk}))


@csrf_exempt
def upload(request, guid_id):

    user = get_object_or_404(User, pk=guid_id)
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            orig_file = request.FILES['docfile']
            orig_image = Image.open(orig_file)
            point = pointillize(image=orig_image, reduce_factor=2)
            point.plotRecPoints(n=40, multiplier=1, fill=False)
            point.plotRandomPointsComplexity(n=2e4, constant=0.01, power=1.3)
            new_stringIO = io.BytesIO()
            point.out.convert('RGB').save(new_stringIO,
                                          orig_file.content_type.split('/')
                                          [-1].upper())
            new_file = InMemoryUploadedFile(new_stringIO,
                                            u"docfile",  # change this?
                                            (orig_file.name.split('.')[0] +
                                             ' pointillized.jpg'),
                                            orig_file.content_type,
                                            None,
                                            None)
            newdoc = user.document_set.create(docfile=new_file)
            newdoc.save()
            origdoc = user.document_set.create(docfile=orig_file)
            origdoc.save()

            # Redirect to the document upload page after POST
            return HttpResponseRedirect(reverse('upload',
                                                kwargs={'guid_id': user.pk}))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the upload page
    all_documents = user.document_set.order_by("-id")
    documents = []
    for document in all_documents:
            if document.docfile.name[-16:] == 'pointillized.jpg':
                documents.append(document)

    # Render upload page with the documents and the form
    return render(
        request,
        'upload.html',
        {'documents': documents, 'form': form, 'guid_id': user.pk}
    )


def gallery(request, guid_id):

    all_documents = Document.objects.order_by("-id")
    documents = []
    for document in all_documents:
        if ((document.docfile.name[-16:] == 'pointillized.jpg') & (document.gallery)):
            documents.append(document)

    return render(request, 'gallery.html', {'documents': documents,
                                            'guid_id': guid_id})


def info(request, guid_id):

    return render(request, 'info.html', {'guid_id': guid_id})


@csrf_exempt
def gif(request, guid_id):

    user = get_object_or_404(User, pk=guid_id)
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            orig_file = request.FILES['docfile']
            orig_image = Image.open(orig_file)
            point = pointillizeStack(image=orig_image, reduce_factor=2,
                                     border=0, queue=True)
            point.crop_Y([1920, 1080], resize=True)
            point.plotRecPoints(n=40, multiplier=1, fill=True)
            point.plotRandomPointsComplexity(n=2e4, constant=0.01, power=1.3)
            multipliers = [5, 4.5, 4, 3.5, 3, 2.6, 2.3, 2, 1.75,
                           1.5, 1.25, 1.1, 1, 1]
            point.build_multipliers(multipliers, reverse=True)
            # point.save_gif('temp/' + guid_id + '_pointqueue.gif', 0.1)
            new_gif_IO = io.BytesIO()
            point.save_gif(new_gif_IO, 0.1)
            new_file = InMemoryUploadedFile(new_gif_IO,
                                            u"docfile",  # change this?
                                            (orig_file.name.split('.')[0] +
                                             ' pointillized.gif'),
                                            'image/gif',
                                            None,
                                            None)
            newdoc = user.document_set.create(docfile=new_file)
            newdoc.save()
            origdoc = user.document_set.create(docfile=orig_file)
            origdoc.save()
            # os.remove('temp/' + guid_id + '_pointqueue.gif')

            # Redirect to the document upload page after POST
            return HttpResponseRedirect(reverse('gif',
                                                kwargs={'guid_id': user.pk}))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the upload page
    all_documents = user.document_set.order_by("-id")
    documents = []
    for document in all_documents:
            if (document.docfile.name[-16:] == 'pointillized.gif'):
                documents.append(document)

    # Render upload page with the documents and the form
    return render(
        request,
        'upload_to_gif.html',
        {'documents': documents, 'form': form, 'guid_id': user.pk}
    )

