# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

from myproject.myapp.models import Document
from myproject.myapp.models import User
from myproject.myapp.forms import DocumentForm

from myproject.myapp.pointillism import pointillize
from PIL import Image
import io
from django.core.files.uploadedfile import InMemoryUploadedFile


def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            user = User()
            user.save()
            orig_file = request.FILES['docfile']
            orig_image = Image.open(orig_file)
            point = pointillize(image=orig_image)
            point.plotRecPoints(step=100, r=100, fill=False)
            point.plotRandomPointsComplexity(n=2e4, constant=0.01, power=1.0)
            new_stringIO = io.BytesIO()
            point.outs[0].convert('RGB').save(new_stringIO,
                                  orig_file.content_type.split('/')[-1].upper())
            new_file = InMemoryUploadedFile(new_stringIO,
                                            u"docfile",  # change this?
                                            'out.jpg',
                                            orig_file.content_type,
                                            None,
                                            None)
            newdoc = user.document_set.create(docfile=new_file)
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('list'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'list.html',
        {'documents': documents, 'form': form}
    )
