# Generated by Django 5.1.4 on 2025-01-15 07:19

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0002_disease_dn'),
    ]

    operations = [
        migrations.AlterField(
            model_name='diagnosis',
            name='image',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='classifier.image'),
        ),
        migrations.AlterField(
            model_name='image',
            name='image',
            field=models.ImageField(upload_to=''),
        ),
    ]
