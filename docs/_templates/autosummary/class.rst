{% block title -%}

.. raw:: html

   <div style="display: None;">

{{ ("``" ~ objname ~ "``") | underline('=')}}

.. raw:: html

   </div>

{%- endblock %}
{% block base %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :member-order: groupwise  {# For consistency with Autosummary #}
    {% if show_inherited_members %}:inherited-members:
    {% endif %}{% if show_undoc_members %}:undoc-members:
    {% endif %}{% if show_inheritance %}:show-inheritance:
    {% endif %}

    {% block attributes %}

    {%- set doc_attributes = [] -%}
    {%- set doc_properties = [] -%}
    {%- for item in attributes -%}
        {%- if show_inherited_members or item not in inherited_members -%}
            {%- if "property" in package[fullname ~ "." ~ item] -%}
                {%- set _ = doc_properties.append(item) -%}
            {%- else -%}
                {%- set _ = doc_attributes.append(item) -%}
            {%- endif -%}
        {%- endif -%}
    {%- endfor %}

    {% if doc_attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
        :nosignatures:
    {% for item in doc_attributes %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}


    {% block methods %}

    {%- set doc_methods = [] -%}
    {%- set doc_constructors = [] -%}
    {%- for item in methods -%}
        {%- if item not in ["__new__", "__init__"] and (show_inherited_members or item not in inherited_members) -%}
            {%- if "constructor" in package[fullname ~ "." ~ item] -%}
                {%- set _ = doc_constructors.append(item) -%}
            {%- else -%}
                {%- set _ = doc_methods.append(item) -%}
            {%- endif -%}
        {%- endif -%}
    {%- endfor %}

    {% if doc_constructors %}
    .. rubric:: {{ _('Constructor Methods') }}

    .. autosummary::
       :nosignatures:
    {% for item in doc_constructors %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}

    {% if doc_methods %}
    .. rubric:: {{ _('Other Methods') }}

    .. autosummary::
        :nosignatures:
    {% for item in doc_methods %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}

    {% endblock %}

    {% if doc_properties %}
    .. rubric:: {{ _('Properties') }}

    .. autosummary::
        :nosignatures:
    {% for item in doc_properties %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

{% endblock %}
