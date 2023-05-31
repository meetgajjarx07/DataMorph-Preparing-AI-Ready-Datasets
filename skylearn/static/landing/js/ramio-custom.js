// ------------------------------------------------
// Project Name: Ramio - Clean Coming Soon and Landing Page Template
// Project Description: Ramio - awesome coming soon template to kick-start your project
// Tags: Ramio, coming soon, under construction, template, coming soon page, html5, css3
// Version: 2.0.1
// Build Date: April 2018
// Last Update: December 2022
// This product is available exclusively on Themeforest
// Author: mix_design
// Author URI: https://themeforest.net/user/mix_design
// File name: ramio-custom.js
// ------------------------------------------------

// ------------------------------------------------
// Table of Contents
// ------------------------------------------------
//
//  1. Loader
//  2. Main Section Loading Animation
//  3. KBW-Countdown
//  4. Mailchimp Notify Form
//  5. Contact Form
//  6. Custom Scrollbar
//  7. YTPlayer
//  8. Vegas Kenburns
//  9. ParticlesJS Backgrounds
//
// ------------------------------------------------
// Table of Contents End
// ------------------------------------------------

$(window).on("load", function() {

  "use strict";

  // --------------------------------------------- //
  // Loader Start
  // --------------------------------------------- //
  setTimeout(function(){
    $(".loader-logo").removeClass('slideInDown').addClass('fadeOutUp');
    $(".loader-caption").removeClass('slideInUp').addClass('fadeOutDown');
  },600);
  // --------------------------------------------- //
  // Loader End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Main Section Loading Animation Start
  // --------------------------------------------- //
  setTimeout(function(){
    $(".loader").addClass('loaded');
    $("#main").addClass('loaded');
  },1200);
  // --------------------------------------------- //
  // Main Section Loading Animation End
  // --------------------------------------------- //

});

$(function() {

  "use strict";

  // --------------------------------------------- //
  // KBW-Countdown Start
  // --------------------------------------------- //
  $('#countdown').countdown({until: $.countdown.UTCDate(+10, 2023, 4, 22), format: 'D'});
  // --------------------------------------------- //
  // KBW-Countdown End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Mailchimp Notify Form Start
  // --------------------------------------------- //
  $('.notify-form').ajaxChimp({
    callback: mailchimpCallback,
    url: 'https://besaba.us10.list-manage.com/subscribe/post?u=e8d650c0df90e716c22ae4778&amp;id=54a7906900'
  });

  function mailchimpCallback(resp) {
    if(resp.result === 'success') {
      $('.notify').find('.form').addClass('is-hidden');
      $('.notify').find('.subscription-ok').addClass('is-visible');
      setTimeout(function() {
        // Done Functions
        $('.notify').find('.subscription-ok').removeClass('is-visible');
        $('.notify').find('.form').delay(300).removeClass('is-hidden');
        $('.notify-form').trigger("reset");
      }, 5000);
    } else if(resp.result === 'error') {
      $('.notify').find('.form').addClass('is-hidden');
      $('.notify').find('.subscription-error').addClass('is-visible');
      setTimeout(function() {
        // Done Functions
        $('.notify').find('.subscription-error').removeClass('is-visible');
        $('.notify').find('.form').delay(300).removeClass('is-hidden');
        $('.notify-form').trigger("reset");
      }, 5000);
    }
  };
  // --------------------------------------------- //
  // Mailchimp Notify Form End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Contact Form Start
  // --------------------------------------------- //
  $("#contact-form").submit(function() { //Change
		var th = $(this);
		$.ajax({
			type: "POST",
			url: "mail.php", //Change
			data: th.serialize()
		}).done(function() {
      $('.writealine').find('.form').addClass('is-hidden');
      $('.writealine').find('.reply-group').addClass('is-visible');
			setTimeout(function() {
				// Done Functions
        $('.writealine').find('.reply-group').removeClass('is-visible');
        $('.writealine').find('.form').delay(300).removeClass('is-hidden');
				th.trigger("reset");
			}, 5000);
		});
		return false;
	});
  // --------------------------------------------- //
  // Contact Form End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Custom Scrollbar Start
  // --------------------------------------------- //
  $(window).on("load resize",function(){
    var selector=".scroll", //your element(s) selector
      cssFlag=window.getComputedStyle(document.querySelector(selector),":after").getPropertyValue("content").replace(/"/g,'');
    if(cssFlag){
      $(selector).mCustomScrollbar();
    }else{
      $(selector).mCustomScrollbar("destroy");
    }
  });
  // --------------------------------------------- //
  // Custom Scrollbar End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // YTPlayer Start
  // --------------------------------------------- //
  var bgndVideo = $("#bgndVideo");
  if(bgndVideo.length){
    bgndVideo.mb_YTPlayer({
      mute: true,
      containment: '#video-wrapper',
      showControls:false,
      autoPlay:true,
      loop:true,
      startAt:0,
      quality:'default'
    });
  }
  // --------------------------------------------- //
  // YTPlayer End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Vegas Kenburns Start
  // --------------------------------------------- //
  //var bgndKenburns = $('#bgndKenburns');
  //if(bgndKenburns.length){
    //bgndKenburns.vegas({
      //timer: false,
      //delay: 8000,
      //transition: 'fade2',
      //transitionDuration: 2000,
      //slides: [
        //{ src: "img/backgrounds/bg-main-color-1.jpg" },
        //{ src: "img/backgrounds/bg-main-color-2.jpg" },
        //{ src: "img/backgrounds/bg-main-color-3.jpg" }
      //],
      //animation: [ 'kenburnsUp', 'kenburnsDown', 'kenburnsLeft', 'kenburnsRight' ]
    //});
  //}
  // --------------------------------------------- //
  // Vegas Kenburns End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Vegas Kenburns Monochrome Start
  // --------------------------------------------- //
  var bgndKenburns = $('#bgndKenburns-monochrome');
  if(bgndKenburns.length){
    bgndKenburns.vegas({
      timer: false,
      delay: 8000,
      transition: 'fade2',
      transitionDuration: 2000,
      slides: [
        { src: "img/backgrounds/bg-main-monochrome_1.jpg" },
        { src: "img/backgrounds/bg-main-monochrome_2.jpg" },
        { src: "img/backgrounds/bg-main-monochrome_3.jpg" }
      ],
      animation: [ 'kenburnsUp', 'kenburnsDown', 'kenburnsLeft', 'kenburnsRight' ]
    });
  }
  // --------------------------------------------- //
  // Vegas Kenburns Monochrome End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Vegas Kenburns Color-1-Light Start
  // --------------------------------------------- //
  var bgndKenburns1 = $('#bgndKenburns-color-1');
  if(bgndKenburns1.length){
    bgndKenburns1.vegas({
      timer: false,
      delay: 8000,
      transition: 'fade2',
      transitionDuration: 2000,
      slides: [
        { src: "img/backgrounds/bg-main-color-1.jpg" },
        { src: "img/backgrounds/bg-main-color-1_1.jpg" },
        { src: "img/backgrounds/bg-main-color-1_2.jpg" }
      ],
      animation: [ 'kenburnsUp', 'kenburnsDown', 'kenburnsLeft', 'kenburnsRight' ]
    });
  }
  // --------------------------------------------- //
  // Vegas Kenburns Color-1-Light End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Vegas Kenburns Color-2 Start
  // --------------------------------------------- //
  var bgndKenburns2 = $('#bgndKenburns-color-2');
  if(bgndKenburns2.length){
    bgndKenburns2.vegas({
      timer: false,
      delay: 8000,
      transition: 'fade2',
      transitionDuration: 2000,
      slides: [
        { src: "img/backgrounds/bg-main-color-2.jpg" },
        { src: "img/backgrounds/bg-main-color-2_1.jpg" },
        { src: "img/backgrounds/bg-main-color-2_2.jpg" }
      ],
      animation: [ 'kenburnsUp', 'kenburnsDown', 'kenburnsLeft', 'kenburnsRight' ]
    });
  }
  // --------------------------------------------- //
  // Vegas Kenburns Color-2 End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Vegas Kenburns Color-3 Start
  // --------------------------------------------- //
  var bgndKenburns3 = $('#bgndKenburns-color-3');
  if(bgndKenburns3.length){
    bgndKenburns3.vegas({
      timer: false,
      delay: 8000,
      transition: 'fade2',
      transitionDuration: 2000,
      slides: [
        { src: "img/backgrounds/bg-main-color-3_1.jpg" },
        { src: "img/backgrounds/bg-main-color-3_2.jpg" },
        { src: "img/backgrounds/bg-main-color-3_3.jpg" }
      ],
      animation: [ 'kenburnsUp', 'kenburnsDown', 'kenburnsLeft', 'kenburnsRight' ]
    });
  }
  // --------------------------------------------- //
  // Vegas Kenburns Color-3 End
  // --------------------------------------------- //

  // --------------------------------------------- //
  // Vegas Kenburns Color-3 Start
  // --------------------------------------------- //
  var bgndKenburns4 = $('#bgndKenburns-color-4');
  if(bgndKenburns4.length){
    bgndKenburns4.vegas({
      timer: false,
      delay: 8000,
      transition: 'fade2',
      transitionDuration: 2000,
      slides: [
        { src: "img/backgrounds/bg-main-color-4_1.jpg" },
        { src: "img/backgrounds/bg-main-color-4_2.jpg" },
        { src: "img/backgrounds/bg-main-color-4_3.jpg" }
      ],
      animation: [ 'kenburnsUp', 'kenburnsDown', 'kenburnsLeft', 'kenburnsRight' ]
    });
  }
  // --------------------------------------------- //
  // Vegas Kenburns Color-3 End
  // --------------------------------------------- //

});

$(window).on("load", function() {

  // --------------------------------------------- //
  // ParticlesJS Backgrounds Start
  // --------------------------------------------- //
  // Night Sky BG - particlesJS
  var bgndNightSky = $('#nightsky-js');
  if (bgndNightSky.length) {
    particlesJS('nightsky-js', {
      "particles": {
        "number": {
          "value": 500,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#ffffff"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.3,
          "random": true,
          "anim": {
            "enable": true,
            "speed": 2,
            "opacity_min": 0.05,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": true,
            "speed": 2,
            "size_min": 1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": false,
          "distance": 150,
          "color": "#ffffff",
          "opacity": 0.4,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": false,
          "attract": {
            "enable": true,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": false,
            "mode": "repulse"
          },
          "onclick": {
            "enable": false,
            "mode": "push"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 400,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 40,
            "duration": 2,
            "opacity": 8,
            "speed": 3
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 4
          },
          "remove": {
            "particles_nb": 2
          }
        }
      },
      "retina_detect": true
    });
  };

  // Particles BG - particlesJS
  var bgndParticles = $('#particles-js');
  if (bgndParticles.length) {
    particlesJS('particles-js', {
        "particles": {
          "number": {
            "value": 80,
            "density": {
              "enable": true,
              "value_area": 800
            }
          },
          "color": {
            "value": "#ffffff"
          },
          "shape": {
            "type": "circle",
            "stroke": {
              "width": 0,
              "color": "#000000"
            },
            "polygon": {
              "nb_sides": 5
            },
            "image": {
              "src": "img/github.svg",
              "width": 100,
              "height": 100
            }
          },
          "opacity": {
            "value": 1,
            "random": true,
            "anim": {
              "enable": true,
              "speed": 1,
              "opacity_min": 0.1,
              "sync": false
            }
          },
          "size": {
            "value": 3,
            "random": true,
            "anim": {
              "enable": false,
              "speed": 80,
              "size_min": 0.1,
              "sync": false
            }
          },
          "line_linked": {
            "enable": true,
            "distance": 180,
            "color": "#ffffff",
            "opacity": 0.5,
            "width": 1
          },
          "move": {
            "enable": true,
            "speed": 3,
            "direction": "none",
            "random": false,
            "straight": false,
            "out_mode": "out",
            "bounce": false,
            "attract": {
              "enable": false,
              "rotateX": 600,
              "rotateY": 1200
            }
          }
        },
        "interactivity": {
          "detect_on": "canvas",
          "events": {
            "onhover": {
              "enable": true,
              "mode": "repulse"
            },
            "onclick": {
              "enable": false,
              "mode": "push"
            },
            "resize": true
          },
          "modes": {
            "grab": {
              "distance": 800,
              "line_linked": {
                "opacity": 1
              }
            },
            "bubble": {
              "distance": 800,
              "size": 80,
              "duration": 2,
              "opacity": 0.8,
              "speed": 3
            },
            "repulse": {
              "distance": 100,
              "duration": 0.4
            },
            "push": {
              "particles_nb": 4
            },
            "remove": {
              "particles_nb": 2
            }
          }
        },
        "retina_detect": true
      });
  };

  // Snow BG - particlesJS
  var bgndSnow = $('#snow-js');
  if (bgndSnow.length) {
      particlesJS('snow-js', {
        "particles": {
          "number": {
            "value": 300,
            "density": {
              "enable": true,
              "value_area": 800
            }
          },
          "color": {
            "value": "#ffffff"
          },
          "shape": {
            "type": "circle",
            "stroke": {
              "width": 0,
              "color": "#000000"
            },
            "polygon": {
              "nb_sides": 5
            },
            "image": {
              "src": "img/github.svg",
              "width": 100,
              "height": 100
            }
          },
          "opacity": {
            "value": 0.4,
            "random": true,
            "anim": {
              "enable": true,
              "speed": 0.2,
              "opacity_min": 0.05,
              "sync": false
            }
          },
          "size": {
            "value": 3,
            "random": true,
            "anim": {
              "enable": false,
              "speed": 2.4,
              "size_min": 3,
              "sync": false
            }
          },
          "line_linked": {
            "enable": false,
            "distance": 150,
            "color": "#ffffff",
            "opacity": 0.4,
            "width": 1
          },
          "move": {
            "enable": true,
            "speed": 1,
            "direction": "bottom",
            "random": true,
            "straight": false,
            "out_mode": "out",
            "bounce": false,
            "attract": {
              "enable": true,
              "rotateX": 600,
              "rotateY": 1200
            }
          }
        },
        "interactivity": {
          "detect_on": "canvas",
          "events": {
            "onhover": {
              "enable": false,
              "mode": "repulse"
            },
            "onclick": {
              "enable": false,
              "mode": "push"
            },
            "resize": true
          },
          "modes": {
            "grab": {
              "distance": 400,
              "line_linked": {
                "opacity": 1
              }
            },
            "bubble": {
              "distance": 400,
              "size": 40,
              "duration": 2,
              "opacity": 8,
              "speed": 3
            },
            "repulse": {
              "distance": 200,
              "duration": 0.4
            },
            "push": {
              "particles_nb": 4
            },
            "remove": {
              "particles_nb": 2
            }
          }
        },
        "retina_detect": true
      });
  };

  // Sunshine Bokeh BG - particlesJS
  var bgndBokeh = $('#bokeh-js');
  if (bgndBokeh.length) {
      particlesJS('bokeh-js', {
        "particles": {
          "number": {
            "value": 24,
            "density": {
              "enable": true,
              "value_area": 800
            }
          },
          "color": {
            "value": "#f08425"
          },
          "shape": {
            "type": "circle",
            "stroke": {
              "width": 0,
              "color": "#000000"
            },
            "polygon": {
              "nb_sides": 5
            },
            "image": {
              "src": "img/github.svg",
              "width": 100,
              "height": 100
            }
          },
          "opacity": {
            "value": 0.2,
            "random": true,
            "anim": {
              "enable": true,
              "speed": 0.24,
              "opacity_min": 0.05,
              "sync": false
            }
          },
          "size": {
            "value": 30,
            "random": true,
            "anim": {
              "enable": false,
              "speed": 40,
              "size_min": 0.1,
              "sync": false
            }
          },
          "line_linked": {
            "enable": false,
            "distance": 150,
            "color": "#ffffff",
            "opacity": 0.4,
            "width": 1
          },
          "move": {
            "enable": true,
            "speed": 0.4,
            "direction": "bottom-left",
            "random": true,
            "straight": false,
            "out_mode": "out",
            "bounce": false,
            "attract": {
              "enable": true,
              "rotateX": 600,
              "rotateY": 1200
            }
          }
        },
        "interactivity": {
          "detect_on": "canvas",
          "events": {
            "onhover": {
              "enable": true,
              "mode": "repulse"
            },
            "onclick": {
              "enable": true,
              "mode": "push"
            },
            "resize": true
          },
          "modes": {
            "grab": {
              "distance": 400,
              "line_linked": {
                "opacity": 1
              }
            },
            "bubble": {
              "distance": 400,
              "size": 40,
              "duration": 2,
              "opacity": 8,
              "speed": 3
            },
            "repulse": {
              "distance": 200,
              "duration": 0.4
            },
            "push": {
              "particles_nb": 4
            },
            "remove": {
              "particles_nb": 2
            }
          }
        },
        "retina_detect": true
      });
  };

  // Triangles BG - particlesJS
  var bgndTriangles = $('#triangles-js');
  if (bgndTriangles.length) {
    particlesJS('triangles-js', {
      "particles": {
        "number": {
          "value": 33,
          "density": {
            "enable": true,
            "value_area": 1420.4657549380909
          }
        },
        "color": {
          "value": "#ffffff"
        },
        "shape": {
          "type": "triangle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.06313181133058181,
          "random": false,
          "anim": {
            "enable": false,
            "speed": 1,
            "opacity_min": 0.1,
            "sync": false
          }
        },
        "size": {
          "value": 11.83721462448409,
          "random": true,
          "anim": {
            "enable": false,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 150,
          "color": "#ffffff",
          "opacity": 0.4,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 6,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": false,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "repulse"
          },
          "onclick": {
            "enable": true,
            "mode": "push"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 400,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 40,
            "duration": 2,
            "opacity": 8,
            "speed": 3
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 4
          },
          "remove": {
            "particles_nb": 2
          }
        }
      },
      "retina_detect": true
    });
  };
  // --------------------------------------------- //
  // ParticlesJS Backgrounds End
  // --------------------------------------------- //

});
