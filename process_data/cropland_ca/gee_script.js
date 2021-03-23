//----------------Earth Engine Code-----------------------
//
// https://code.earthengine.google.com/
// Create a geodesic polygon containing Central Valley, CA
// and Mexicali region of Baja California
//
//----------------Earth Engine Code-----------------------

var aoi1 = ee.Geometry.Polygon([
  [[-122.5, 40], [-121.5, 40], [-121.5, 39], [-122.5, 39]]
]);
var aoi2 = ee.Geometry.Polygon([
  [[-122, 39], [-121, 39], [-121, 38], [-122, 38] ]
]);
var aoi3 = ee.Geometry.Polygon([
  [[-121.5, 38], [-120.5, 38], [-120.5, 37.5], [-121.5, 37.5] ]
]);
var aoi32 = ee.Geometry.Polygon([
  [[-121, 37.5], [-120, 37.5], [-120, 37], [-121, 37] ]
]);
var aoi4 = ee.Geometry.Polygon([
  [[-120.75, 37], [-119.75, 37], [-119.75, 36.5], [-120.75, 36.5] ]
]);
var aoi5 = ee.Geometry.Polygon([
  [[-120.15, 36.5], [-119.15, 36.5], [-119.15, 36], [-120.15, 36] ]
]);
var aoi6 = ee.Geometry.Polygon([
  [[-119.5, 36], [-119, 36], [-119, 35.5], [-119.5, 35.5] ]
]);
var aoi7 = ee.Geometry.Polygon([
  [[-119.5, 35.5], [-118.75, 35.5], [-118.75, 35], [-119.5, 35] ]
]);

var aoi8 = ee.Geometry.Polygon([
  [[-115.250, 32.75], [-115.25, 32.5], [-115, 32.5], [-115, 32.75]]
]);
var aoi81 = ee.Geometry.Polygon([
  [[-115.50, 33], [-115.5, 32.75], [-115.25, 32.75], [-115.25, 33]]
]);

var aoi82 = ee.Geometry.Polygon([
  [[-115.750, 33.25], [-115.75, 33], [-115.5, 33], [-115.5, 33.25]]
]);

var aoi83 = ee.Geometry.Polygon([
  [[-116, 33], [-116, 32.75], [-115.75, 32.75], [-115.75, 33]]
]);

var aoi84 = ee.Geometry.Polygon([
  [[-116, 32.75], [-116, 32.5], [-115.75, 32.5], [-115.75, 32.75]]
]);

// Display the polygon on the map
//Map.centerObject(boulder);
Map.addLayer(aoi1, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi2, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi3, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi32, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi4, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi5, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi6, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi7, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi8, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi81, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi82, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi83, {color: 'limegreen'}, 'geodesic polygon');
Map.addLayer(aoi84, {color: 'limegreen'}, 'geodesic polygon');
