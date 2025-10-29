{% block title scoped -%}

{%- if fullname.startswith("openff") and fullname.count(".") == 1 -%}
{%- set title = fullname -%}
{%- else -%}
{%- set title = objname -%}
{%- endif -%}
{{ ("``" ~ title ~ "``") | underline('=')}}

{%- endblock %}

{% block base scoped %}
{%- set doc_functions = [] -%}
{%- set doc_attributes = [] -%}
{%- set doc_exceptions = [] -%}
{%- set doc_modules = [] -%}
{%- set doc_classes = [] -%}
{%- set doc_others = [] -%}
{%- for item in members -%}
    {%- if not item.startswith('_') -%}
        {%- set item_info = package[fullname ~ "." ~ item] -%}
        {%- if item in exceptions or "exception" in item_info -%}
            {%- set _ = doc_exceptions.append(item) -%}
        {%- elif item in functions or "function" in item_info -%}
            {%- set _ = doc_functions.append(item) -%}
        {%- elif fullname ~ "." ~ item in modules or "module" in item_info -%}
            {%- set _ = doc_modules.append(item) -%}
        {%- elif item in classes or "class" in item_info -%}
            {%- set _ = doc_classes.append(item) -%}
        {%- elif item in attributes or "attribute" in item_info -%}
            {%- set _ = doc_attributes.append(item) -%}
        {%- else -%}
            {%- set _ = doc_others.append(item) -%}
        {%- endif -%}
   {%- endif -%}
{%- endfor %}


.. automodule:: {{ fullname }}

   {% if doc_modules %}
   .. rubric:: Modules
        :name: {{fullname}}:modules

   .. autosummary::
      :caption: Modules
      :toctree:
      :recursive:
   {% for item in doc_modules %}
   {%- if item not in exclude_modules %}
      ~{{ item }}
   {%- endif -%}
   {%- endfor %}

   {% endif %}

   {% if doc_attributes %}
   .. rubric:: {{ _('Module Attributes') }}
        :name: {{fullname}}:attributes

   .. autosummary::
      :caption: Attributes
      :toctree:
      :nosignatures:
   {% for item in doc_attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}

   {% if doc_functions %}
   .. rubric:: {{ _('Functions') }}
        :name: {{fullname}}:functions

   .. autosummary::
      :caption: Functions
      :toctree:
      :nosignatures:
   {% for item in doc_functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}


   {% if doc_classes %}
   .. rubric:: {{ _('Classes') }}
        :name: {{fullname}}:classes

   .. autosummary::
      :caption: Classes
      :toctree:
      :nosignatures:
   {% for item in doc_classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}

   {% if doc_exceptions %}
   .. rubric:: {{ _('Exceptions') }}
        :name: {{fullname}}:exceptions

   .. autosummary::
      :caption: Exceptions
      :toctree:
      :nosignatures:
   {% for item in doc_exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}


   {% if doc_others %}
   .. rubric:: {{ _('Other Objects') }}
        :name: {{fullname}}:others

   .. autosummary::
      :caption: Other Objects
      :toctree:
      :nosignatures:
   {% for item in doc_others %}
      {{ item }}
   {%- endfor %}
   {% endif %}

{% endblock %}
