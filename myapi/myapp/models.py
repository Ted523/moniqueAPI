from django.db import models

class Car(models.Model):
    name = models.CharField(max_length=100)
    top_speed = models.IntegerField()

from pandas import options


class location(models.Model):

    County= models.TextField(max_length=300)
    Total_x = models.CharField(max_length=300)
    Male_x = models.CharField(max_length=300)
    Female_x= models.CharField(max_length=300)
    Total_y= models.CharField(max_length=300)
    Male_y= models.CharField(max_length=300)
    Female_Y= models.CharField(max_length=300)
    Intersex= models.CharField(max_length=300)
    Population= models.CharField(max_length=300)
    Land= models.CharField(max_length=300)
    ppdens= models.CharField(max_length=300)
    

    class Meta:
        abstract = False


    def _str_(self):
        return 'County : {0} .Population: {1] Land :{2}'.format(self.County, self.Land, self.Population)
