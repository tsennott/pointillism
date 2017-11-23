from django.contrib import admin

# Register your models here.
from myproject.myapp.models import Document
from myproject.myapp.models import User

admin.site.register(User)


class DocumentAdmin(admin.ModelAdmin):
    fields = ["docfile"]
    list_display = ("docfile", "image_img", "gallery")
    list_editable = ["gallery"]


admin.site.register(Document, DocumentAdmin)
