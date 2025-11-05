(chapter:ccdcache)=
# Customizing CCD access

In [](chapter:reslib), we discussed a bunch of ways to augment the CCD with your own residues, or to construct a fully custom residue library of your own. Sometimes these options might not cut it, and you'll want to take full control of how the CCD is accessed. The [`CcdCache`] class lets you do this. The following features that you might have taken for granted can be controlled by creating your own `CcdCache`:

1. The location of the on-disk cache, with the [`cache_path`] argument
2. One or several paths to pre-stocked library directories of CCD `.CIF` files, such as those distributed with Pablo. You might use this to package more or different default residue definitions with your application or library. Controlled with the [`library_paths`] argument.
3. The patches you want to apply to the CCD before exposing definitions to Pablo. Pablo comes with quite a few patches to fix bugs and inconsistencies we've found in the CCD, but maybe you disagree with our fix --- the [`patches`] argument lets you specify your own tree of patch functions.

There are a few other arguments too, but they can be controlled more easily with the existing API. You can read the [API docs] to get more details on how a custom `CcdCache` is created.

[API docs]: openff.pablo.ccd.CcdCache
[`CcdCache`]: openff.pablo.ccd.CcdCache
[`cache_path`]: openff.pablo.ccd.CcdCache.params.cache_path
[`library_paths`]: openff.pablo.ccd.CcdCache.params.library_paths
[`patches`]: openff.pablo.ccd.CcdCache.params.patches
