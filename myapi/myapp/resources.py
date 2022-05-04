from import_export import resources
from .models import location

class collectiveResource(resources.ModelResource):
    class Meta:
        model = location