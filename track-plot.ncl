
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"


begin

start_1=23

  EXP = (/"EXP_I"/)                
  nexp = dimsizes(EXP)

  a = addfile("wrfout_d03_2018-06-09_18:00:00.nc","r")
  b = addfile("track-RTG-mp_physics_2-cu_physics_1.nc","r")


  lat2d = a->XLAT(0,:,:)
  lon2d = a->XLONG(0,:,:)
  dimll = dimsizes(lat2d)
  nlat  = dimll(0)
  mlon  = dimll(1)

	SLP_series = b->SLP_series
        INTENSITY_series = b->INTENSITY_series
	imin = b->imin
	jmin = b->jmin
	time = b->time

	ndate=dimsizes(time)
	ndate_2=dimsizes(imin)

	print(ndate)
	print(ndate_2)



  wks=gsn_open_wks("X11","track")             
  gsn_define_colormap(wks,"BlGrYeOrReVi200") 

  res                     = True
  res@gsnDraw             = False             
  res@gsnFrame            = False             
  res@gsnMaximize         = True             

  res@tiMainString = "Hurricane Bud"      
  WRF_map_c(a,res,0)                          
                                             
  plot = gsn_csm_map(wks,res)                

; Set up resources for polymarkers.
  gsres                = True
  gsres@gsMarkerIndex  = 16                  
  ;gsres@gsMarkerSizeF = 0.005              


; Set up resources for polylines.
  res_lines                      = True
  res_lines@gsLineThicknessF     = 3.           ; 3x as thick

 

; Loop through each date and add polylines to the plot.
  do i = start_1, ndate-2
     res_lines@gsLineColor           = cols(1)   ;;;;;;;;;;;
     xx=(/lon2d(imin(i),jmin(i)),lon2d(imin(i+1),jmin(i+1))/)
     yy=(/lat2d(imin(i),jmin(i)),lat2d(imin(i+1),jmin(i+1))/)
     line(i) = gsn_add_polyline(wks,plot,xx,yy,res_lines)
  end do

;  lon1d = ndtooned(lon2d)
;  lat1d = ndtooned(lat2d)

; Loop through each date and add polymarkers to the plot.
  do i =start_1, ndate-1
     print("dot:"+lon2d(imin(i),jmin(i))+","+lat2d(imin(i),jmin(i)))
     gsres@gsMarkerColor  = cols(4) ;;;;;
     dot(i)=gsn_add_polymarker(wks,plot,lon2d(imin(i),jmin(i)),lat2d(imin(i),jmin(i)),gsres)
  end do

;  do i =start_1, ndate-1
;     print("dot:"+lon1d(smin(i))+","+lat1d(smin(i)))
;     gsres@gsMarkerColor  = cols(0)
;     dot(i)=gsn_add_polymarker(wks,plot,lon1d(smin(i)),lat1d(smin(i)),gsres)
;  end do


; Date (Legend)
  txres               = True
  txres@txFontHeightF = 0.015
  txres@txFontColor   = cols(0)

  txid1 = new(ndate,graphic)

  do i = 0, ndate-1
     txres@txJust = "CenterRight"
     ix = smin(i) - 4
     print("Eye:"+ix)
     if(i.eq.1) then
        txres@txJust = "CenterLeft"
        ix = ix + 8
     end if
     txid1(i) = gsn_add_text(wks,plot,sdate(i),lon1d(ix),lat1d(ix),txres)
  end do

  txres@txJust = "CenterLeft"
  txid2 = new(nexp,graphic)
  pmid2 = new(nexp,graphic)
  do i = 0,nexp-1
     gsres@gsMarkerColor  = cols(i)
     txres@txFontColor    = cols(i)
     ii = ((/129,119,109/))  ; ilat
     jj = ((/110,110,110/))  ; jlon
     ji = ii*mlon+jj         ; col x row
     pmid2(i) = gsn_add_polymarker(wks,plot,lon1d(ji(i)),lat1d(ji(i)),gsres)
     txid2(i) = gsn_add_text(wks,plot,EXP(i),lon1d(ji(i)+5),lat1d(ji(i)),txres)
  end do

  draw(plot)
  frame(wks)
end

